from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from analysis_common import (
    FEATURES7, FEATURES8, MODEL_FINAL, MODEL_REDUCED,
    add_patient_bootstrap_intervals,
    binary_metrics, fit_pipeline, normalize_common_fields,
    prediction_frame, raw_scale_coefficients, repeated_nested_oof,
    validate_analysis_table,
)


def run_internal_validation(
    mimic_csv: Path,
    public_dir: Path,
    private_dir: Path,
    cfg: Dict,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(mimic_csv)
    validate_analysis_table(df, "mimic")
    df = normalize_common_fields(df)
    df = df[df["outcome72"].notna()].copy()
    df["outcome72"] = df["outcome72"].astype(int)

    models = [
        (MODEL_REDUCED, FEATURES7),
        (MODEL_FINAL, FEATURES8),
    ]
    summary_rows = []
    fitted = {}
    pred_frames = []

    for model_name, features in models:
        p_oof, selected_cs = repeated_nested_oof(
            df,
            features=features,
            c_grid=cfg["c_grid"],
            outer_folds=int(cfg["outer_folds"]),
            outer_repeats=int(cfg["outer_repeats"]),
            inner_folds=int(cfg["inner_folds"]),
            seed=int(cfg["random_seed"]),
        )
        m = binary_metrics(df["outcome72"], p_oof)
        m["model"] = model_name
        m["selected_c_median"] = float(np.median(selected_cs))
        m["selected_c_mode"] = float(pd.Series(selected_cs).mode().iloc[0])
        m = add_patient_bootstrap_intervals(
            m, df["outcome72"], p_oof,
            n_boot=int(cfg["bootstrap_iterations"]),
            seed=int(cfg["random_seed"]) + 100,
        )

        # Compatibility aliases used by the retained final figure script.
        m["roc_auc_ci_low"] = m["roc_auc_patient_boot_ci_low"]
        m["roc_auc_ci_high"] = m["roc_auc_patient_boot_ci_high"]
        summary_rows.append(m)

        pred_frames.append(
            prediction_frame(
                df, p_oof,
                analysis="MIMIC_repeated_5x5_nested_CV",
                model_name=model_name,
            )
        )

        final_pipe = fit_pipeline(
            df[list(features)],
            df["outcome72"],
            C=float(cfg["final_c"]),
        )
        fitted[model_name] = final_pipe
        joblib.dump(final_pipe, private_dir / f"{model_name}_full_mimic.joblib")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        public_dir / "94_MIMIC七八变量模型内部验证汇总.csv",
        index=False, encoding="utf-8-sig"
    )

    # Export final model formula and preprocessing parameters.
    formula_parts = []
    for model_name, features in models:
        coef = raw_scale_coefficients(fitted[model_name], features)
        coef["model"] = model_name
        formula_parts.append(coef)
    formula = pd.concat(formula_parts, ignore_index=True)
    formula.to_csv(
        public_dir / "98_最终模型公式与预处理参数.csv",
        index=False, encoding="utf-8-sig"
    )

    pd.concat(pred_frames, ignore_index=True).to_csv(
        private_dir / "internal_patient_predictions.csv",
        index=False, encoding="utf-8-sig"
    )

    return df, fitted
