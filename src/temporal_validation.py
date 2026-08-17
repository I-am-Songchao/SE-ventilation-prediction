from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from analysis_common import (
    FEATURES7, FEATURES8, MODEL_FINAL, MODEL_REDUCED,
    add_patient_bootstrap_intervals, binary_metrics, fit_pipeline,
    prediction_frame,
)


def run_temporal_validation(
    mimic_df: pd.DataFrame,
    public_dir: Path,
    private_dir: Path,
    cfg: Dict,
) -> pd.DataFrame:
    df = mimic_df.copy()
    df["vent_start_year"] = pd.to_numeric(df["vent_start_year"], errors="coerce")

    train = df[df["vent_start_year"].between(2008, 2016)].copy()
    test = df[df["vent_start_year"].between(2017, 2022)].copy()

    if train.empty or test.empty:
        raise ValueError("Temporal split is empty. Check vent_start_year.")

    rows = []
    pred_frames = []

    for model_name, features in [
        (MODEL_REDUCED, FEATURES7),
        (MODEL_FINAL, FEATURES8),
    ]:
        pipe = fit_pipeline(
            train[list(features)],
            train["outcome72"].astype(int),
            C=float(cfg["final_c"]),
        )
        p = pipe.predict_proba(test[list(features)])[:, 1]
        m = binary_metrics(test["outcome72"].astype(int), p)
        m.update({
            "model": model_name,
            "n_train": len(train),
            "n_train_event": int(train["outcome72"].sum()),
        })
        m = add_patient_bootstrap_intervals(
            m, test["outcome72"], p,
            n_boot=int(cfg["bootstrap_iterations"]),
            seed=int(cfg["random_seed"]) + 200,
        )
        rows.append(m)
        pred_frames.append(
            prediction_frame(
                test, p,
                analysis="MIMIC_2008_2016_to_2017_2022",
                model_name=model_name,
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(
        public_dir / "106_MIMIC七八变量时间验证汇总.csv",
        index=False, encoding="utf-8-sig"
    )
    pred = pd.concat(pred_frames, ignore_index=True)
    pred.to_csv(
        private_dir / "temporal_patient_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    return pred
