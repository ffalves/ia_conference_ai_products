import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def generate_metadata(
    filename: str,
    status: str,
    source: str,
    contract_version: str,
    report: Optional[Dict] = None,
    artifacts: Optional[Dict[str, Optional[Path]]] = None,
) -> Path:
    """
    Gera um arquivo JSON com metadados sobre os dados processados.

    Parâmetros
    ----------
    filename : str
        Nome do arquivo principal de saída (ex.: valid_data_20250807_101500.csv)
    status : str
        "valid" | "invalid" | "neutral"
    source : str
        Nome do arquivo de origem (ex.: propensao_compra.csv)
    contract_version : str
        Versão do contrato aplicado
    report : dict, opcional
        Relatório retornado pelo enforcer (rows_total, rows_valid, rows_invalid, invalid_pct, status, failure_details...)
    artifacts : dict, opcional
        Caminhos de artefatos gerados: ex. {"valid_csv": Path, "invalid_csv": Path, "invalid_json": Path}

    Retorna
    -------
    Path
        Caminho do arquivo de metadados gerado.
    """
    assert status in {"valid", "invalid", "neutral"}, "status deve ser 'valid', 'invalid' ou 'neutral'"

    # Base: raw_layer/data/catalog/metadata_<status>/
    base_path = Path(__file__).resolve().parent.parent / "data" / "catalog"
    output_dir = base_path / f"metadata_{status}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")

    # Normaliza artefatos (converte Path -> str para serializar)
    artifacts_serialized = None
    if artifacts:
        artifacts_serialized = {
            k: (str(v) if v is not None else None)
            for k, v in artifacts.items()
        }

    # Report enxuto (evita jogar failure_details completo aqui para não inflar o arquivo)
    report_summary = None
    if report:
        report_summary = {
            "batch_status": report.get("status"),
            "rows_total": report.get("rows_total"),
            "rows_valid": report.get("rows_valid"),
            "rows_invalid": report.get("rows_invalid"),
            "invalid_pct": report.get("invalid_pct"),
            "contract_version": report.get("contract_version"),
        }

    metadata = {
        "filename": filename,
        "status": status,
        "source_file": source,
        "contract_version": contract_version,
        "timestamp": ts.isoformat(),
        "artifacts": artifacts_serialized,
        "report": report_summary,
    }

    output_file = output_dir / f"metadata_{status}_{ts_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[METADATA] Metadado gerado em: {output_file}")
    return output_file
