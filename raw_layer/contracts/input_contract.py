# raw_layer/contracts/input_contract.py
import re
import json
import yaml
import argparse
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from pandera.errors import SchemaErrors

# -------------------------------------------------
# Constantes / helpers
# -------------------------------------------------
EMAIL_REGEX = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
CPF_REGEX   = r"^\d{3}\.\d{3}\.\d{3}-\d{2}$"
USER_ID_REGEX = r"^U\d{3}$"


def is_valid_cpf(cpf_str: str) -> bool:
    if not isinstance(cpf_str, str):
        return False
    if not re.match(CPF_REGEX, cpf_str):
        return False
    nums = re.sub(r"\D", "", cpf_str)
    if len(nums) != 11 or nums == nums[0] * 11:
        return False

    def calc_digito(n: str) -> str:
        soma = sum((len(n) + 1 - i) * int(x) for i, x in enumerate(n))
        resto = soma % 11
        return "0" if resto < 2 else str(11 - resto)

    d1 = calc_digito(nums[:9])
    d2 = calc_digito(nums[:9] + d1)
    return nums[-2:] == d1 + d2


def sha256_of_file(path: Optional[Union[str, Path]]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_contract(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect_unique_keys(model: Dict[str, Any]) -> List[str]:
    uniques: List[str] = []
    if "unique_keys" in model and isinstance(model["unique_keys"], list):
        uniques.extend(model["unique_keys"])
    for fld in model.get("fields", []):
        for q in (fld.get("quality") or []):
            if str(q.get("type")).lower() == "uniqueness":
                uniques.append(fld["name"])
    return list(dict.fromkeys(uniques))


# -------------------------------------------------
# Checks de campo (respeitando modo dos customizados)
# -------------------------------------------------
def _build_checks_for_field(field: Dict[str, Any], *, custom_mode: str) -> List[Check]:
    """
    custom_mode: 'hard' | 'soft' | 'off'
    - hard: adiciona checks customizados ao Pandera (derrubam registro)
    - soft: NÃO adiciona no Pandera (serão avaliados após e apenas reportados)
    - off : ignora checks customizados
    """
    checks: List[Check] = []
    constraints = field.get("constraints", {}) or {}
    enum = field.get("enum")

    # min/max
    if "min" in constraints:
        checks.append(Check.ge(constraints["min"]))
    if "max" in constraints:
        checks.append(Check.le(constraints["max"]))

    # domínio
    if enum:
        checks.append(Check.isin(enum))

    # pattern explícito em constraints
    patt = constraints.get("pattern")
    if patt:
        checks.append(Check.str_matches(patt))

    # formato conhecido
    fmt = field.get("format")
    if fmt == "email":
        checks.append(Check.str_matches(EMAIL_REGEX))

    # quality -> regex (sempre hard) / custom (depende do modo)
    for q in (field.get("quality") or []):
        qtype = str(q.get("type", "")).lower()
        if qtype == "regex" and q.get("pattern"):
            checks.append(Check.str_matches(q["pattern"]))
        elif qtype == "custom":
            name = str(q.get("name", "")).lower()
            if name == "cpf_checksum":
                if custom_mode == "hard":
                    checks.append(Check(lambda s: s.astype(str).apply(is_valid_cpf), element_wise=False))
                # soft/off: não adiciona aqui (vamos calcular depois)
    return checks


def _mask_value(val: Any) -> Any:
    if val is None:
        return None
    s = str(val)
    if "@" in s and re.match(EMAIL_REGEX, s):
        user, _, dom = s.partition("@")
        return f"{user[:2]}***@{dom}"
    if re.match(CPF_REGEX, s):
        return f"{s[:3]}.***.***-{s[-2:]}"
    return s[0] + "*" * max(0, len(s) - 2) + (s[-1] if len(s) > 1 else "")


def _mask_failure_details(failure_df: Optional[pd.DataFrame], mask_fields: List[str]) -> pd.DataFrame:
    if failure_df is None or failure_df.empty:
        return pd.DataFrame()
    out = failure_df.copy()
    if "column" in out.columns and "failure_case" in out.columns:
        m = out["column"].isin(mask_fields)
        out.loc[m, "failure_case"] = out.loc[m, "failure_case"].apply(_mask_value)
    return out


# -------------------------------------------------
# Schema a partir do contrato
# -------------------------------------------------
def build_schema(contract: Dict[str, Any]) -> Tuple[DataFrameSchema, str]:
    # modo dos checks customizados (default: hard para compatibilidade antiga)
    validation = contract.get("validation", {}) or {}
    custom_mode = str(validation.get("custom_checks_enforcement", "hard")).lower()
    if custom_mode not in {"hard", "soft", "off"}:
        custom_mode = "hard"

    try:
        model = contract["models"]["users"]
        fields = model["fields"]
    except KeyError:
        model = {"fields": contract.get("fields", [])}
        fields = model["fields"]

    unique_keys = _collect_unique_keys(model)

    type_map = {
        "string":  pa.String,
        "text":    pa.String,
        "integer": pa.Int,
        "float":   pa.Float,
        "boolean": pa.Bool,
        "bool":    pa.Bool,
    }

    columns: Dict[str, Column] = {}
    for field in fields:
        name = field["name"]
        dtype = str(field.get("type", "string")).lower()
        required = bool(field.get("required", True))
        if dtype not in type_map:
            raise ValueError(f"Tipo desconhecido no contrato: '{dtype}' (campo: {name})")

        col_type = type_map[dtype]
        checks = _build_checks_for_field(field, custom_mode=custom_mode)
        col_unique = name in unique_keys

        columns[name] = Column(
            col_type,
            checks=checks,
            required=required,
            nullable=not required,
            unique=col_unique,
            coerce=True,
        )

    return DataFrameSchema(columns), custom_mode


# -------------------------------------------------
# Observabilidade / relatório
# -------------------------------------------------
def _observability_report(
    df: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_invalid: pd.DataFrame,
    contract: Dict[str, Any],
    failure_cases_masked: pd.DataFrame,
    batch_id: str,
    source_path: Path,
    soft_custom_diag: Dict[str, Any],
) -> Dict[str, Any]:
    total = len(df)
    invalid_pct = 0.0 if total == 0 else round(len(df_invalid) * 100.0 / total, 2)

    obs = (contract.get("observability") or {})
    acc = (obs.get("acceptance_criteria") or {})
    warn_thr = acc.get("batch_warn_if_invalid_pct_gt")
    fail_thr = acc.get("batch_fail_if_invalid_pct_gt")
    per_field_max_null_pct = acc.get("per_field_max_null_pct")
    required_fields_presence = (acc.get("required_fields_presence") or {})

    status = "accepted"
    if fail_thr is not None and invalid_pct > float(fail_thr):
        status = "rejected"
    elif warn_thr is not None and invalid_pct > float(warn_thr):
        status = "accepted_with_warning"

    presence_checks = {}
    for col, min_pct in required_fields_presence.items():
        if col in df.columns:
            filled = df[col].notna().sum()
            pct = 0.0 if total == 0 else round(filled * 100.0 / total, 2)
            presence_checks[col] = {"observed_pct": pct, "required_pct": float(min_pct)}
        else:
            presence_checks[col] = {"observed_pct": 0.0, "required_pct": float(min_pct), "note": "coluna ausente no lote"}

    null_ratio = {col: (0.0 if total == 0 else round(df[col].isna().sum() * 100.0 / total, 2)) for col in df.columns}

    null_ratio_alerts = {}
    if per_field_max_null_pct is not None:
        lim = float(per_field_max_null_pct)
        for col, pct in null_ratio.items():
            if pct > lim:
                null_ratio_alerts[col] = {"observed_null_pct": pct, "limit": lim}

    domain_viol = {}
    if not failure_cases_masked.empty and "column" in failure_cases_masked.columns:
        domain_viol = failure_cases_masked["column"].value_counts().to_dict()

    report = {
        "batch_id": batch_id,
        "source_path": str(source_path),
        "source_sha256": sha256_of_file(source_path),
        "rows_total": total,
        "rows_valid": int(len(df_valid)),
        "rows_invalid": int(len(df_invalid)),
        "invalid_records_pct": invalid_pct,
        "status": status,
        "contract_version": contract.get("info", {}).get("version", "unknown"),
        "required_fields_presence": presence_checks,
        "null_ratio_per_field": null_ratio,
        "null_ratio_alerts": null_ratio_alerts,
        "domain_violations_per_field": domain_viol,
        "failure_details": failure_cases_masked.to_dict(orient="records"),
        # novidades:
        "soft_custom_checks": soft_custom_diag or {},
    }
    return report


def _write_report_files(report: Dict[str, Any], outdir: Path) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = outdir / f"load_report_{ts}.json"
    csv_path  = outdir / f"load_report_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    rows = [
        {"metric": "batch_id", "value": report["batch_id"]},
        {"metric": "source_path", "value": report["source_path"]},
        {"metric": "source_sha256", "value": report.get("source_sha256")},
        {"metric": "rows_total", "value": report["rows_total"]},
        {"metric": "rows_valid", "value": report["rows_valid"]},
        {"metric": "rows_invalid", "value": report["rows_invalid"]},
        {"metric": "invalid_records_pct", "value": report["invalid_records_pct"]},
        {"metric": "status", "value": report["status"]},
        {"metric": "contract_version", "value": report["contract_version"]},
    ]
    # presença requerida
    for col, d in report.get("required_fields_presence", {}).items():
        rows.append({"metric": f"required_presence_{col}", "value": d.get("observed_pct"), "details": json.dumps(d, ensure_ascii=False)})
    # null ratio alerts
    for col, d in report.get("null_ratio_alerts", {}).items():
        rows.append({"metric": f"null_ratio_alert_{col}", "value": d.get("observed_null_pct"), "details": json.dumps(d, ensure_ascii=False)})
    # domínio
    if report.get("domain_violations_per_field"):
        rows.append({"metric": "domain_violations_per_field", "value": "", "details": json.dumps(report["domain_violations_per_field"], ensure_ascii=False)})
    # soft custom checks
    if report.get("soft_custom_checks"):
        rows.append({"metric": "soft_custom_checks", "value": "", "details": json.dumps(report["soft_custom_checks"], ensure_ascii=False)})

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


# -------------------------------------------------
# Metadados
# -------------------------------------------------
def _write_metadata(
    status: str,
    count: int,
    source_file: Path,
    contract_version: str,
    out_root: Path,
    batch_id: str,
    sample_output_file: Optional[Path] = None,
) -> Path:
    assert status in {"valid", "invalid"}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = out_root / "catalog" / (f"metadata_{status}")
    base_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": status,
        "batch_id": batch_id,
        "row_count": int(count),
        "source_file": str(source_file),
        "source_sha256": sha256_of_file(source_file),
        "contract_version": contract_version,
        "timestamp": datetime.now().isoformat(),
        "sample_output_file": str(sample_output_file) if sample_output_file else None,
        "sample_output_sha256": sha256_of_file(sample_output_file) if sample_output_file else None,
        "producer": "input_contract_enforcer",
    }
    out_path = base_dir / f"metadata_{status}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[METADATA] ({status}) {out_path}")
    return out_path


# -------------------------------------------------
# Validação + segmentação + relatório (SEM quebrar)
# -------------------------------------------------
def validate_with_contract(
    df: pd.DataFrame,
    contract_path: str,
    *,
    batch_id: Optional[str],
    source_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    contract = load_contract(contract_path)
    schema, custom_mode = build_schema(contract)

    pii_mask_fields: List[str] = (
        contract.get("security", {}).get("pii_handling", {}).get("mask_in_logs", []) or []
    )

    failure_cases: Optional[pd.DataFrame] = None
    try:
        validated = schema.validate(df, lazy=True)
        df_valid = validated.copy()
        df_invalid = pd.DataFrame(columns=df.columns)
    except SchemaErrors as e:
        failure_cases = e.failure_cases
        # índice pode vir como "row_index" (mais comum) ou "index" dependendo da versão
        idx_col = "row_index" if "row_index" in failure_cases.columns else ("index" if "index" in failure_cases.columns else None)
        if idx_col is not None:
            idx_series = pd.to_numeric(failure_cases[idx_col], errors="coerce").dropna().astype(int)
            bad_idx = pd.Index(idx_series.unique())
            df_invalid = df.loc[bad_idx].copy()
            df_valid = df.drop(index=bad_idx, errors="ignore").copy()
        else:
            df_invalid = df.copy()
            df_valid = pd.DataFrame(columns=df.columns)

    # --- Soft custom checks (apenas diagnóstico; NÃO derruba registros) ---
    soft_custom_diag: Dict[str, Any] = {}
    if custom_mode == "soft":
        # CPF checksum
        if "cpf" in df.columns:
            mask_invalid_cpf = ~df["cpf"].astype(str).apply(is_valid_cpf)
            soft_custom_diag["cpf_checksum"] = {
                "violations_count": int(mask_invalid_cpf.sum()),
                "violations_idx_sample": df.index[mask_invalid_cpf].tolist()[:20],
            }

    masked_failures = _mask_failure_details(failure_cases, pii_mask_fields) if isinstance(failure_cases, pd.DataFrame) else pd.DataFrame()
    report = _observability_report(df, df_valid, df_invalid, contract, masked_failures, batch_id, source_path, soft_custom_diag)
    return df_valid, df_invalid, report


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Input Contract Enforcer (não quebra execução; gera relatórios e metadados)")
    parser.add_argument("--csv", required=True, help="Caminho do CSV de entrada")
    parser.add_argument("--contract", required=True, help="Caminho do contrato YAML")
    parser.add_argument("--out-root", default="raw_layer/data", help="Raiz de saída (default: raw_layer/data)")
    parser.add_argument("--show-invalid", action="store_true", help="Mostra amostra dos inválidos no stdout")
    args = parser.parse_args()

    in_csv = Path(args.csv)
    contract_path = Path(args.contract)
    out_root = Path(args.out_root)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]

    df_in = pd.read_csv(in_csv)
    df_valid, df_invalid, report = validate_with_contract(
        df_in, str(contract_path), batch_id=batch_id, source_path=in_csv
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    valid_dir   = out_root / "processed_raw_data" / "valid"
    logs_dir    = out_root / "logs"
    reports_dir = out_root / "catalog" / "load_reports"

    valid_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    valid_path = valid_dir / f"valid_data_{ts}.csv"
    invalid_path = logs_dir / f"invalid_records_{ts}.csv"

    if not df_valid.empty:
        df_valid.to_csv(valid_path, index=False)
    else:
        valid_path = None

    if not df_invalid.empty:
        df_invalid.to_csv(invalid_path, index=False)
    else:
        invalid_path = None

    json_path, csv_path = _write_report_files(report, reports_dir)

    contract_version = report.get("contract_version", "unknown")
    _write_metadata(
        status="valid",
        count=len(df_valid),
        source_file=in_csv,
        contract_version=contract_version,
        out_root=out_root,
        batch_id=batch_id,
        sample_output_file=valid_path,
    )
    _write_metadata(
        status="invalid",
        count=len(df_invalid),
        source_file=in_csv,
        contract_version=contract_version,
        out_root=out_root,
        batch_id=batch_id,
        sample_output_file=invalid_path,
    )

    print("\n=== ENFORCER REPORT ===")
    print(f"Batch ID : {batch_id}")
    print(f"Contrato : {contract_path}")
    print(f"CSV      : {in_csv}")
    print(f"Source SHA256: {sha256_of_file(in_csv)}")
    print(f"Total    : {report['rows_total']}")
    print(f"Válidos  : {report['rows_valid']}")
    print(f"Inválidos: {report['rows_invalid']} ({report['invalid_records_pct']}%)")
    req = report.get("required_fields_presence", {})
    if req:
        print("Presenças obrigatórias:")
        for k, v in req.items():
            print(f"  - {k}: {v.get('observed_pct')}% (mín.: {v.get('required_pct')})")
    print(f"Status   : {report['status']}")
    print(f"Versão   : {report.get('contract_version','unknown')}")
    print(f"Relatório JSON: {json_path}")
    print(f"Relatório CSV : {csv_path}")

    if args.show_invalid and not df_invalid.empty:
        print("\n--- Amostra de inválidos ---")
        print(df_invalid.head(10))


# -------------------------------------------------
# Uso programático (compatível com testes e pipeline)
# -------------------------------------------------
def run_enforcer(
    csv_path: Union[str, Path],
    contract_path: Union[str, Path],
    valid_dir: Union[str, Path],
    invalid_dir: Union[str, Path],
    report_json_path: Union[str, Path],
    show_invalid_sample: bool = False,
) -> Dict[str, Any]:
    """
    Executa o enforcer programaticamente (sem CLI), usando os diretórios passados (posicionais ou nomeados).
    Retorna um dicionário com 'report' e 'paths' importantes.
    """
    csv_path = Path(csv_path)
    contract_path = Path(contract_path)
    valid_dir = Path(valid_dir)
    invalid_dir = Path(invalid_dir)
    report_json_path = Path(report_json_path)

    df_in = pd.read_csv(csv_path)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]

    df_valid, df_invalid, report = validate_with_contract(
        df_in, str(contract_path), batch_id=batch_id, source_path=csv_path
    )

    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    valid_path = valid_dir / f"valid_data_{ts}.csv"
    invalid_path = invalid_dir / f"invalid_records_{ts}.csv"

    if not df_valid.empty:
        df_valid.to_csv(valid_path, index=False)
    else:
        valid_path = None

    if not df_invalid.empty:
        df_invalid.to_csv(invalid_path, index=False)
    else:
        invalid_path = None

    # histórico timestampado + arquivo estável
    reports_dir = report_json_path.parent / "load_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamped_json, stamped_csv = _write_report_files(report, reports_dir)

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    report_csv_path = report_json_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {"metric": "batch_id", "value": report["batch_id"]},
            {"metric": "source_path", "value": report["source_path"]},
            {"metric": "source_sha256", "value": report.get("source_sha256")},
            {"metric": "rows_total", "value": report["rows_total"]},
            {"metric": "rows_valid", "value": report["rows_valid"]},
            {"metric": "rows_invalid", "value": report["rows_invalid"]},
            {"metric": "invalid_records_pct", "value": report["invalid_records_pct"]},
            {"metric": "status", "value": report["status"]},
            {"metric": "contract_version", "value": report["contract_version"]},
        ]
    ).to_csv(report_csv_path, index=False)

    # metadados (raiz comum .../data)
    out_root = report_json_path.parent.parent
    contract_version = report.get("contract_version", "unknown")
    _write_metadata(
        status="valid",
        count=len(df_valid),
        source_file=csv_path,
        contract_version=contract_version,
        out_root=out_root,
        batch_id=batch_id,
        sample_output_file=valid_path,
    )
    _write_metadata(
        status="invalid",
        count=len(df_invalid),
        source_file=csv_path,
        contract_version=contract_version,
        out_root=out_root,
        batch_id=batch_id,
        sample_output_file=invalid_path,
    )

    if show_invalid_sample and invalid_path is not None:
        print("\n--- Amostra de inválidos ---")
        print(df_invalid.head(10))

    return {
        "report": report,
        "paths": {
            "valid_csv": valid_path,
            "invalid_csv": invalid_path,
            "report_json": report_json_path,
            "report_csv": report_csv_path,
            "stamped_report_json": stamped_json,
            "stamped_report_csv": stamped_csv,
        },
        "batch_id": report["batch_id"],
        "status": report["status"],
    }


if __name__ == "__main__":
    main()
