from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


EXPECTED = {
    "MIMIC development n": 393,
    "MIMIC development events": 156,
    "MIMIC temporal validation n": 147,
    "MIMIC temporal validation events": 60,
    "eICU primary target n": 146,
    "eICU primary target events": 31,
    "eICU strict GCS-observed n": 81,
    "eICU strict GCS-observed events": 19,
    "MIMIC repeated CV AUROC 8-variable": 0.760,
    "MIMIC temporal AUROC 8-variable": 0.687,
    "eICU strict AUROC 8-variable": 0.708,
}


def run_checklist(
    mimic_df: pd.DataFrame,
    eicu_df: pd.DataFrame,
    public_dir: Path,
) -> pd.DataFrame:
    from analysis_common import MODEL_FINAL
    from external_validation_and_sensitivity import primary_target

    internal = pd.read_csv(public_dir / "94_MIMIC七八变量模型内部验证汇总.csv")
    temporal = pd.read_csv(public_dir / "106_MIMIC七八变量时间验证汇总.csv")
    external = pd.read_csv(public_dir / "99_eICU原始外部验证性能汇总.csv")

    target = primary_target(eicu_df)
    strict = target[target["gcs_motor_landmark"].notna()].copy()

    actual = {
        "MIMIC development n": int(len(mimic_df)),
        "MIMIC development events": int(mimic_df["outcome72"].sum()),
        "MIMIC temporal validation n": int((mimic_df["vent_start_year"].between(2017, 2022)).sum()),
        "MIMIC temporal validation events": int(
            mimic_df.loc[mimic_df["vent_start_year"].between(2017, 2022), "outcome72"].sum()
        ),
        "eICU primary target n": int(len(target)),
        "eICU primary target events": int(target["outcome72"].sum()),
        "eICU strict GCS-observed n": int(len(strict)),
        "eICU strict GCS-observed events": int(strict["outcome72"].sum()),
        "MIMIC repeated CV AUROC 8-variable": float(
            internal.loc[internal["model"] == MODEL_FINAL, "roc_auc"].iloc[0]
        ),
        "MIMIC temporal AUROC 8-variable": float(
            temporal.loc[temporal["model"] == MODEL_FINAL, "roc_auc"].iloc[0]
        ),
        "eICU strict AUROC 8-variable": float(
            external.loc[
                (external["dataset"] == "primary_within24h_confirmed_or_probable")
                & (external["model"] == MODEL_FINAL),
                "roc_auc",
            ].iloc[0]
        ),
    }

    rows = []
    for k, expected in EXPECTED.items():
        observed = actual.get(k, np.nan)
        if isinstance(expected, int):
            match = observed == expected
        else:
            match = abs(float(observed) - expected) <= 0.01
        rows.append({
            "check": k,
            "manuscript_expected": expected,
            "reconstructed_observed": observed,
            "match_within_rule": bool(match),
        })
    out = pd.DataFrame(rows)
    out.to_csv(public_dir / "reconstruction_checklist.csv", index=False, encoding="utf-8-sig")
    return out
