# raw_layer/tests/test_pipeline_input.py
import sys
import json
from pathlib import Path
import pandas as pd

# garante import do pacote local
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_LAYER = PROJECT_ROOT / "raw_layer"
sys.path.append(str(RAW_LAYER))

from contracts.input_contract import run_enforcer  # noqa: E402

MIN_CONTRACT_YAML = r"""
dataContractSpecification: 1.0.0
id: input_contract_test
info:
  title: Input Contract (Test)
  version: 0.0.1
security:
  pii_handling:
    mask_in_logs: ["nome","email","cpf"]
observability:
  acceptance_criteria:
    batch_warn_if_invalid_pct_gt: 50
    batch_fail_if_invalid_pct_gt: 90
    required_fields_presence:
      email: 75
    per_field_max_null_pct: 50
models:
  users:
    type: object
    primary_key: ["user_id"]
    unique_keys: ["user_id","email"]
    fields:
      - name: user_id
        type: string
        required: true
        quality:
          - type: regex
            pattern: '^U\d{3}$'
          - type: uniqueness
            level: dataset
      - name: nome
        type: string
        required: true
      - name: email
        type: string
        required: true
        quality:
          - type: regex
            pattern: '^[^@\s]+@[^@\s]+\.[^@\s]+$'
          - type: uniqueness
            level: dataset
      - name: idade
        type: integer
        required: true
        constraints: { min: 18, max: 100 }
      - name: canal_preferido
        type: string
        required: true
        enum: ["app","site","whatsapp","email"]
"""

def test_pipeline_input_end_to_end(tmp_path: Path):
    # 1) prepara árvore de diretórios tmp
    base = tmp_path / "raw_layer" / "data"
    raw_dir = base / "raw_data"
    valid_dir = base / "processed_raw_data" / "valid"
    logs_dir = base / "logs"
    catalog_dir = base / "catalog"
    raw_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    # 2) cria CSV pequeno com 4 linhas (2 válidas, 2 inválidas)
    df = pd.DataFrame([
        {"user_id":"U001","nome":"Ana Souza","email":"ana@example.com","idade":28,"canal_preferido":"app"},
        {"user_id":"U002","nome":"Bruno Lima","email":"bruno@example.org","idade":35,"canal_preferido":"email"},
        {"user_id":"XYZ","nome":"Carlos","email":"carlos@example.com","idade":40,"canal_preferido":"site"},
        {"user_id":"U004","nome":"Dana","email":None,"idade":17,"canal_preferido":"telefone"},
    ])[["user_id","nome","email","idade","canal_preferido"]]

    in_csv = raw_dir / "raw_data.csv"
    df.to_csv(in_csv, index=False)

    # 3) grava contrato mínimo
    contract_path = tmp_path / "raw_layer" / "contracts" / "input_contract.yaml"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(MIN_CONTRACT_YAML, encoding="utf-8")

    # 4) define caminho do relatório
    report_json_path = catalog_dir / "load_report.json"

    # 5) executa enforcer
    run_enforcer(
    csv_path=in_csv,
    contract_path=contract_path,
    valid_dir=valid_dir,
    invalid_dir=logs_dir,
    report_json_path=report_json_path,
    )

    # 6) valida artefatos
    valid_files = list(valid_dir.glob("valid_data_*.csv"))
    assert len(valid_files) == 1
    df_valid_out = pd.read_csv(valid_files[0])
    assert len(df_valid_out) == 2
    assert set(df_valid_out["user_id"]) == {"U001","U002"}

    invalid_files = list(logs_dir.glob("invalid_records_*.csv"))
    assert len(invalid_files) == 1
    df_invalid_out = pd.read_csv(invalid_files[0])
    assert len(df_invalid_out) == 2
    assert set(df_invalid_out["user_id"]) == {"XYZ","U004"}

    report_jsons = list(catalog_dir.rglob("load_report*.json"))
    report_csvs  = list(catalog_dir.rglob("load_report*.csv"))
    assert report_jsons
    assert report_csvs

    report = json.loads(report_jsons[0].read_text(encoding="utf-8"))
    assert report["rows_total"] == 4
    assert report["rows_valid"] == 2
    assert report["rows_invalid"] == 2
    assert report["status"] in {"accepted","accepted_with_warning","rejected"}

    md_valid = list(catalog_dir.rglob("metadata_valid/metadata_valid_*.json"))
    md_invalid = list(catalog_dir.rglob("metadata_invalid/metadata_invalid_*.json"))
    assert md_valid
    assert md_invalid
