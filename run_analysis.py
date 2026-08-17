from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from analysis_common import normalize_common_fields, validate_analysis_table
from model_development_and_internal_validation import run_internal_validation
from temporal_validation import run_temporal_validation
from external_validation_and_sensitivity import prepare_eicu, run_external_validation
from reconstruction_checklist import run_checklist


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    public_dir = ROOT / cfg["public_results_dir"]
    private_dir = ROOT / cfg["private_results_dir"]
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    mimic_csv = Path(cfg["mimic_analysis_table"])
    eicu_csv = Path(cfg["eicu_analysis_table"])

    print("[1/4] Repeated internal validation and final frozen MIMIC models")
    mimic_df, fitted = run_internal_validation(
        mimic_csv, public_dir, private_dir, cfg
    )

    print("[2/4] Temporal validation")
    temporal_pred = run_temporal_validation(
        mimic_df, public_dir, private_dir, cfg
    )

    print("[3/4] eICU external validation and sensitivity analyses")
    external = run_external_validation(
        eicu_csv, public_dir, private_dir, cfg
    )

    # Combine private prediction files into manuscript-compatible plotting input.
    internal_path = private_dir / "internal_patient_predictions.csv"
    temporal_path = private_dir / "temporal_patient_predictions.csv"
    external_path = private_dir / "external_patient_predictions.csv"
    parts = []
    for p in [internal_path, temporal_path, external_path]:
        if p.exists():
            parts.append(pd.read_csv(p))
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(
            private_dir / "112_第六阶段患者级预测.csv",
            index=False, encoding="utf-8-sig"
        )

    print("[4/4] Reconstruction checklist")
    eicu_df = prepare_eicu(eicu_csv)
    check = run_checklist(mimic_df, eicu_df, public_dir)
    print(check.to_string(index=False))

    print("\nAnalysis completed.")
    print("Public aggregate outputs:", public_dir)
    print("PRIVATE patient-level outputs:", private_dir)
    print(
        "\nImportant: do not describe this reconstruction as the exact original "
        "analysis code until the checklist is verified against the manuscript."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
