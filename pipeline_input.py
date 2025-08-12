# pipeline_input.py
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE / "raw_layer"))

# importa a função utilitária do enforcer
from contracts.input_contract import run_enforcer


def main():
    csv_path      = BASE / "raw_layer" / "data" / "raw_data" / "raw_data.csv"
    contract_path = BASE / "raw_layer" / "contracts" / "input_contract.yaml"

    out_valid_dir   = BASE / "raw_layer" / "data" / "processed_raw_data" / "valid"
    out_invalid_dir = BASE / "raw_layer" / "data" / "logs"
    report_json     = BASE / "raw_layer" / "data" / "catalog" / "load_report.json"

    print(f"[PIPELINE] Executando enforcer sobre: {csv_path}")
    result = run_enforcer(
        csv_path=csv_path,
        contract_path=contract_path,
        valid_dir=out_valid_dir,
        invalid_dir=out_invalid_dir,
        report_json_path=report_json,
        show_invalid_sample=False,  # mude para True se quiser ver amostra no stdout
    )

    print(
        f"[PIPELINE] Concluído | status={result['report']['status']} "
        f"| total={result['report']['rows_total']} "
        f"| valid={result['report']['rows_valid']} "
        f"| invalid={result['report']['rows_invalid']} "
        f"| report_json={result['paths']['report_json']}"
    )
    print("Input pipeline concluído.")


if __name__ == "__main__":
    main()
