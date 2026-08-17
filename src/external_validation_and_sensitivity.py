from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from analysis_common import (
    FEATURES7, FEATURES8, MODEL_FINAL, MODEL_REDUCED,
    add_cluster_bootstrap_intervals, add_patient_bootstrap_intervals,
    apply_full_recalibration, apply_intercept_update, binary_metrics,
    intercept_only_update, normalize_common_fields, prediction_frame,
    validate_analysis_table,
)


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).astype(bool)
    x = s.astype(str).str.strip().str.lower()
    return x.isin(["1", "true", "yes", "y"])


def prepare_eicu(eicu_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(eicu_csv)
    validate_analysis_table(df, "eicu")
    df = normalize_common_fields(df)

    for c in ["primary_within24h", "eligible_landmark", "postanoxic_excluded"]:
        df[c] = _bool_series(df[c])

    df["ventilation_class"] = df["ventilation_class"].astype(str).str.strip().str.lower()
    df["se_first_offset_min"] = pd.to_numeric(df["se_first_offset_min"], errors="coerce")
    df = df[df["outcome72"].notna()].copy()
    df["outcome72"] = df["outcome72"].astype(int)
    return df


def primary_target(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["primary_within24h"])
        & (df["eligible_landmark"])
        & (~df["postanoxic_excluded"])
        & (df["ventilation_class"].isin(["confirmed", "probable"]))
    ].copy()


def score_frozen(
    df: pd.DataFrame,
    pipe,
    features: List[str],
    analysis: str,
    model_name: str,
) -> pd.DataFrame:
    p = pipe.predict_proba(df[features])[:, 1]
    return prediction_frame(df, p, analysis=analysis, model_name=model_name)


def summarize_external(
    pred: pd.DataFrame,
    dataset_name: str,
    cfg: Dict,
) -> Dict[str, float]:
    y = pred["outcome"].astype(int)
    p = pred["predicted_probability"].astype(float)
    row = binary_metrics(y, p)
    row["dataset"] = dataset_name
    row["model"] = pred["model"].iloc[0]
    row["n_hospitals"] = int(pred["hospitalid"].nunique())
    row = add_patient_bootstrap_intervals(
        row, y, p,
        n_boot=int(cfg["bootstrap_iterations"]),
        seed=int(cfg["random_seed"]) + 300,
    )
    row = add_cluster_bootstrap_intervals(
        row, pred,
        n_boot=int(cfg["cluster_bootstrap_iterations"]),
        seed=int(cfg["random_seed"]) + 400,
    )
    return row


def run_external_validation(
    eicu_csv: Path,
    public_dir: Path,
    private_dir: Path,
    cfg: Dict,
) -> pd.DataFrame:
    df = prepare_eicu(eicu_csv)
    target = primary_target(df)

    pipe8 = joblib.load(private_dir / f"{MODEL_FINAL}_full_mimic.joblib")
    pipe7 = joblib.load(private_dir / f"{MODEL_REDUCED}_full_mimic.joblib")

    # Strict primary external-validation subgroup: observed GCS required.
    strict = target[target["gcs_motor_landmark"].notna()].copy()

    pred8 = score_frozen(
        strict, pipe8, FEATURES8,
        analysis="primary_GCS_observed_same_subset",
        model_name=MODEL_FINAL,
    )
    pred7 = score_frozen(
        strict, pipe7, FEATURES7,
        analysis="primary_GCS_observed_same_subset",
        model_name=MODEL_REDUCED,
    )

    rows = [
        summarize_external(pred8, "primary_within24h_confirmed_or_probable", cfg),
        summarize_external(pred7, "primary_within24h_confirmed_or_probable", cfg),
    ]

    # Cohort-definition sensitivity analyses; all require observed GCS.
    sensitivity_defs = [
        (
            "sensitivity_within24h_confirmed_only",
            target[
                (target["ventilation_class"] == "confirmed")
                & target["gcs_motor_landmark"].notna()
            ],
        ),
        (
            "sensitivity_landmark_SE_confirmed_or_probable",
            target[
                (target["se_first_offset_min"] <= 360)
                & target["gcs_motor_landmark"].notna()
            ],
        ),
        (
            "sensitivity_within24h_all_ventilation_evidence",
            df[
                (df["primary_within24h"])
                & (df["eligible_landmark"])
                & (~df["postanoxic_excluded"])
                & df["gcs_motor_landmark"].notna()
            ],
        ),
        (
            "sensitivity_anytime_SE_confirmed_or_probable",
            df[
                (df["eligible_landmark"])
                & (~df["postanoxic_excluded"])
                & (df["ventilation_class"].isin(["confirmed", "probable"]))
                & df["gcs_motor_landmark"].notna()
            ],
        ),
    ]

    sensitivity_preds = []
    for name, cohort in sensitivity_defs:
        if cohort.empty:
            continue
        p = score_frozen(cohort, pipe8, FEATURES8, analysis=name, model_name=MODEL_FINAL)
        rows.append(summarize_external(p, name, cfg))
        sensitivity_preds.append(p)

    # Complete target cohort with frozen MIMIC median imputation, including GCS.
    # SimpleImputer inside the frozen pipeline applies the MIMIC median.
    complete_pred = score_frozen(
        target, pipe8, FEATURES8,
        analysis="primary_all_patients_MIMIC_median_imputation",
        model_name=MODEL_FINAL,
    )
    rows.append(summarize_external(
        complete_pred,
        "complete_target_GCS_median_imputed",
        cfg,
    ))

    out = pd.DataFrame(rows)
    out.to_csv(
        public_dir / "99_eICU原始外部验证性能汇总.csv",
        index=False, encoding="utf-8-sig"
    )

    # Descriptive recalibration in strict primary external validation.
    original = pred8.copy()
    y = original["outcome"].astype(int).to_numpy()
    p = original["predicted_probability"].astype(float).to_numpy()
    delta = intercept_only_update(y, p)
    from analysis_common import calibration_intercept_slope
    ci, cs = calibration_intercept_slope(y, p)
    p_i = apply_intercept_update(p, delta)
    p_full = apply_full_recalibration(p, ci, cs)
    from sklearn.metrics import brier_score_loss
    recal = pd.DataFrame([{
        "model": MODEL_FINAL,
        "cohort": "primary_strict_eICU_external_validation",
        "original_brier": brier_score_loss(y, p),
        "intercept_only_update_amount": delta,
        "brier_after_intercept_only_update": brier_score_loss(y, p_i),
        "intercept_slope_update_intercept": ci,
        "intercept_slope_update_slope": cs,
        "brier_after_full_update": brier_score_loss(y, p_full),
    }])
    recal.to_csv(
        public_dir / "eICU_descriptive_recalibration.csv",
        index=False, encoding="utf-8-sig"
    )

    # Hospital-level GCS documentation coverage in the primary target cohort.
    g = target.copy()
    g["gcs_observed"] = g["gcs_motor_landmark"].notna().astype(int)
    coverage = (
        g.groupby("hospital_id", as_index=False)
         .agg(
            n=("patient_id", "size"),
            n_outcome=("outcome72", "sum"),
            n_combined_gcs=("gcs_observed", "sum"),
         )
    )
    coverage["combined_coverage_pct"] = 100 * coverage["n_combined_gcs"] / coverage["n"]
    coverage["cohort_definition"] = "SE_within24h_no_postanoxic"
    coverage = coverage.rename(columns={"hospital_id": "hospitalid"})
    coverage.to_csv(
        public_dir / "77_eICU医院层面GCS覆盖率.csv",
        index=False, encoding="utf-8-sig"
    )

    # Center-restriction validation.
    center_rows = []
    restrictions = [
        ("GCS_coverage_100_and_hospital_n_ge2", 100.0, True),
        ("GCS_coverage_ge80_and_hospital_n_ge2", 80.0, True),
        ("GCS_coverage_ge80_no_minimum_hospital_n", 80.0, False),
    ]
    for label, threshold, require_n2 in restrictions:
        cov = coverage[coverage["combined_coverage_pct"] >= threshold].copy()
        if require_n2:
            cov = cov[cov["n"] >= 2]
        hospitals = set(cov["hospitalid"])
        cohort = strict[strict["hospital_id"].isin(hospitals)].copy()
        if cohort.empty:
            continue
        psub = score_frozen(
            cohort, pipe8, FEATURES8,
            analysis=label, model_name=MODEL_FINAL
        )
        row = summarize_external(psub, label, cfg)
        row["center_restriction"] = label
        center_rows.append(row)
    pd.DataFrame(center_rows).to_csv(
        public_dir / "111_eICU按医院GCS覆盖限制验证.csv",
        index=False, encoding="utf-8-sig"
    )

    # Patient-level file is PRIVATE. It exists only to regenerate ROC/calibration/S1 locally.
    all_preds = [pred7, pred8, complete_pred] + sensitivity_preds
    all_pred = pd.concat(all_preds, ignore_index=True)
    all_pred.to_csv(
        private_dir / "external_patient_predictions.csv",
        index=False, encoding="utf-8-sig"
    )

    # A manuscript-compatible private prediction export.
    all_pred.to_csv(
        private_dir / "112_第六阶段患者级预测.csv",
        index=False, encoding="utf-8-sig"
    )

    return out
