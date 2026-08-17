from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES8 = [
    "age",
    "sex",
    "gcs_motor_landmark",
    "map_min",
    "spo2_min",
    "fio2_max",
    "peep_max",
    "bun_latest",
]

FEATURES7 = [
    "age",
    "sex",
    "map_min",
    "spo2_min",
    "fio2_max",
    "peep_max",
    "bun_latest",
]

MODEL_FINAL = "Ridge8_GCS"
MODEL_REDUCED = "Ridge7_harmonized"


def clip_probability(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_probability(p)
    return np.log(p / (1 - p))


def validate_analysis_table(df: pd.DataFrame, dataset: str) -> None:
    base = {
        "patient_id", "hospital_id", "age", "sex", "gcs_motor_landmark",
        "map_min", "spo2_min", "fio2_max", "peep_max", "bun_latest", "outcome72"
    }
    if dataset == "mimic":
        base.add("vent_start_year")
    if dataset == "eicu":
        base.update({
            "ventilation_class", "se_first_offset_min", "primary_within24h",
            "eligible_landmark", "postanoxic_excluded"
        })
    missing = sorted(base.difference(df.columns))
    if missing:
        raise ValueError(f"{dataset}: missing required columns: {missing}")


def normalize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["age"] = pd.to_numeric(x["age"], errors="coerce").clip(upper=90)

    # Accept common encodings for sex but standardize to female=1, male=0.
    if x["sex"].dtype == object:
        s = x["sex"].astype(str).str.strip().str.lower()
        sex_map = {
            "female": 1, "f": 1, "1": 1,
            "male": 0, "m": 0, "0": 0,
        }
        x["sex"] = s.map(sex_map)
    x["sex"] = pd.to_numeric(x["sex"], errors="coerce")

    numeric_cols = [
        "gcs_motor_landmark", "map_min", "spo2_min",
        "fio2_max", "peep_max", "bun_latest", "outcome72",
    ]
    for c in numeric_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # Harmonize FiO2 if percentages were accidentally supplied.
    mask_pct = x["fio2_max"] > 1.5
    x.loc[mask_pct, "fio2_max"] = x.loc[mask_pct, "fio2_max"] / 100.0

    # Manuscript-matched plausible ranges.
    x.loc[~x["gcs_motor_landmark"].between(1, 6), "gcs_motor_landmark"] = np.nan
    x.loc[~x["map_min"].between(20, 250), "map_min"] = np.nan
    x.loc[~x["spo2_min"].between(40, 100), "spo2_min"] = np.nan
    x.loc[~x["fio2_max"].between(0.21, 1.00), "fio2_max"] = np.nan
    x.loc[~x["peep_max"].between(0, 30), "peep_max"] = np.nan
    x.loc[~x["bun_latest"].between(1, 300), "bun_latest"] = np.nan

    x["outcome72"] = x["outcome72"].astype("Int64")
    return x


def make_pipeline(C: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=float(C),
                    solver="liblinear",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )


def select_c(
    X: pd.DataFrame,
    y: pd.Series,
    c_grid: Sequence[float],
    inner_folds: int,
    random_state: int,
) -> float:
    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=make_pipeline(1.0),
        param_grid={"model__C": list(c_grid)},
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    return float(grid.best_params_["model__C"])


def fit_pipeline(X: pd.DataFrame, y: pd.Series, C: float) -> Pipeline:
    pipe = make_pipeline(C)
    pipe.fit(X, y)
    return pipe


def calibration_intercept_slope(
    y: Sequence[int],
    p: Sequence[float],
) -> Tuple[float, float]:
    y_arr = np.asarray(y, dtype=float)
    lp = logit(np.asarray(p, dtype=float))
    if len(np.unique(y_arr)) < 2:
        return np.nan, np.nan
    X = sm.add_constant(lp, has_constant="add")
    try:
        model = sm.GLM(y_arr, X, family=sm.families.Binomial()).fit()
        return float(model.params[0]), float(model.params[1])
    except Exception:
        return np.nan, np.nan


def intercept_only_update(
    y: Sequence[int],
    p: Sequence[float],
) -> float:
    y_arr = np.asarray(y, dtype=float)
    lp = logit(np.asarray(p, dtype=float))
    if len(np.unique(y_arr)) < 2:
        return np.nan
    offset = lp
    X = np.ones((len(y_arr), 1), dtype=float)
    try:
        fit = sm.GLM(
            y_arr, X, family=sm.families.Binomial(), offset=offset
        ).fit()
        return float(fit.params[0])
    except Exception:
        return np.nan


def apply_intercept_update(p: Sequence[float], delta: float) -> np.ndarray:
    lp = logit(np.asarray(p, dtype=float)) + float(delta)
    return 1 / (1 + np.exp(-lp))


def apply_full_recalibration(
    p: Sequence[float], intercept: float, slope: float
) -> np.ndarray:
    lp = float(intercept) + float(slope) * logit(np.asarray(p, dtype=float))
    return 1 / (1 + np.exp(-lp))


def binary_metrics(y: Sequence[int], p: Sequence[float]) -> Dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = clip_probability(np.asarray(p, dtype=float))
    out: Dict[str, float] = {
        "n": int(len(y_arr)),
        "n_event": int(y_arr.sum()),
        "event_rate": float(y_arr.mean()) if len(y_arr) else np.nan,
        "mean_predicted_risk": float(p_arr.mean()) if len(p_arr) else np.nan,
    }
    if len(np.unique(y_arr)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_arr, p_arr))
        out["pr_auc"] = float(average_precision_score(y_arr, p_arr))
        out["log_loss"] = float(log_loss(y_arr, p_arr, labels=[0, 1]))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
        out["log_loss"] = np.nan
    out["brier"] = float(brier_score_loss(y_arr, p_arr))
    prevalence = float(y_arr.mean()) if len(y_arr) else np.nan
    null_brier = prevalence * (1 - prevalence) if np.isfinite(prevalence) else np.nan
    out["scaled_brier"] = (
        1 - out["brier"] / null_brier if null_brier and null_brier > 0 else np.nan
    )
    ci, cs = calibration_intercept_slope(y_arr, p_arr)
    out["calibration_intercept"] = ci
    out["calibration_slope"] = cs
    out["oe_ratio"] = (
        float(y_arr.mean() / p_arr.mean()) if p_arr.mean() > 0 else np.nan
    )
    return out


def bootstrap_metric_ci(
    y: Sequence[int],
    p: Sequence[float],
    metric: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    vals: List[float] = []
    n = len(y_arr)
    if n == 0:
        return np.nan, np.nan

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y_arr[idx]
        pp = p_arr[idx]
        if metric in {"roc_auc", "pr_auc"} and len(np.unique(yy)) < 2:
            continue
        try:
            if metric == "roc_auc":
                val = roc_auc_score(yy, pp)
            elif metric == "pr_auc":
                val = average_precision_score(yy, pp)
            elif metric == "brier":
                val = brier_score_loss(yy, pp)
            elif metric == "calibration_intercept":
                val = calibration_intercept_slope(yy, pp)[0]
            elif metric == "calibration_slope":
                val = calibration_intercept_slope(yy, pp)[1]
            elif metric == "oe_ratio":
                val = yy.mean() / pp.mean()
            else:
                raise ValueError(metric)
            if np.isfinite(val):
                vals.append(float(val))
        except Exception:
            continue

    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cluster_bootstrap_metric_ci(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    cluster_col: str,
    metric: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = pd.Series(df[cluster_col].dropna().unique())
    if clusters.empty:
        return np.nan, np.nan
    vals: List[float] = []

    grouped = {c: g.copy() for c, g in df.groupby(cluster_col)}
    cluster_values = clusters.to_numpy()

    for _ in range(n_boot):
        sampled = rng.choice(cluster_values, size=len(cluster_values), replace=True)
        parts = []
        # Repeatedly sampled hospitals are treated as independent bootstrap clusters.
        for j, c in enumerate(sampled):
            g = grouped[c].copy()
            g["_boot_cluster"] = j
            parts.append(g)
        b = pd.concat(parts, ignore_index=True)
        yy = b[y_col].astype(int).to_numpy()
        pp = b[p_col].astype(float).to_numpy()
        if metric in {"roc_auc", "pr_auc"} and len(np.unique(yy)) < 2:
            continue
        try:
            if metric == "roc_auc":
                val = roc_auc_score(yy, pp)
            elif metric == "pr_auc":
                val = average_precision_score(yy, pp)
            elif metric == "brier":
                val = brier_score_loss(yy, pp)
            elif metric == "calibration_intercept":
                val = calibration_intercept_slope(yy, pp)[0]
            elif metric == "calibration_slope":
                val = calibration_intercept_slope(yy, pp)[1]
            elif metric == "oe_ratio":
                val = yy.mean() / pp.mean()
            else:
                raise ValueError(metric)
            if np.isfinite(val):
                vals.append(float(val))
        except Exception:
            continue

    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def repeated_nested_oof(
    df: pd.DataFrame,
    features: Sequence[str],
    c_grid: Sequence[float],
    outer_folds: int,
    outer_repeats: int,
    inner_folds: int,
    seed: int,
) -> Tuple[np.ndarray, List[float]]:
    X = df[list(features)]
    y = df["outcome72"].astype(int).to_numpy()

    rkf = RepeatedStratifiedKFold(
        n_splits=outer_folds, n_repeats=outer_repeats, random_state=seed
    )
    pred_sum = np.zeros(len(df), dtype=float)
    pred_n = np.zeros(len(df), dtype=int)
    selected_cs: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test = X.iloc[test_idx]

        c = select_c(
            X_train,
            pd.Series(y_train),
            c_grid=c_grid,
            inner_folds=inner_folds,
            random_state=seed + fold_idx,
        )
        selected_cs.append(c)
        model = fit_pipeline(X_train, pd.Series(y_train), C=c)
        p = model.predict_proba(X_test)[:, 1]

        pred_sum[test_idx] += p
        pred_n[test_idx] += 1

    if np.any(pred_n == 0):
        raise RuntimeError("Some observations did not receive out-of-fold predictions.")
    return pred_sum / pred_n, selected_cs


def raw_scale_coefficients(pipe: Pipeline, features: Sequence[str]) -> pd.DataFrame:
    imputer: SimpleImputer = pipe.named_steps["imputer"]
    scaler: StandardScaler = pipe.named_steps["scaler"]
    model: LogisticRegression = pipe.named_steps["model"]

    med = np.asarray(imputer.statistics_, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    beta_std = np.asarray(model.coef_[0], dtype=float)
    intercept_std = float(model.intercept_[0])

    beta_raw = beta_std / scale
    intercept_raw = intercept_std - np.sum(beta_std * mean / scale)

    rows = [{
        "term": "intercept",
        "raw_coefficient": intercept_raw,
        "standardized_coefficient": np.nan,
        "imputation_median": np.nan,
        "scaler_mean": np.nan,
        "scaler_scale": np.nan,
    }]
    for i, f in enumerate(features):
        rows.append({
            "term": f,
            "raw_coefficient": beta_raw[i],
            "standardized_coefficient": beta_std[i],
            "imputation_median": med[i],
            "scaler_mean": mean[i],
            "scaler_scale": scale[i],
        })
    return pd.DataFrame(rows)


def prediction_frame(
    source_df: pd.DataFrame,
    p: np.ndarray,
    analysis: str,
    model_name: str,
) -> pd.DataFrame:
    cols = [
        "patient_id", "hospital_id", "outcome72",
        "gcs_motor_landmark", "age", "sex", "map_min",
        "spo2_min", "fio2_max", "peep_max", "bun_latest",
    ]
    out = source_df[[c for c in cols if c in source_df.columns]].copy()
    out = out.rename(columns={"outcome72": "outcome", "hospital_id": "hospitalid"})
    out["predicted_probability"] = p
    out["analysis"] = analysis
    out["model"] = model_name
    return out


def add_patient_bootstrap_intervals(
    row: Dict[str, float], y: Sequence[int], p: Sequence[float],
    n_boot: int, seed: int
) -> Dict[str, float]:
    for metric in [
        "roc_auc", "pr_auc", "brier",
        "calibration_intercept", "calibration_slope", "oe_ratio"
    ]:
        low, high = bootstrap_metric_ci(y, p, metric, n_boot, seed + hash(metric) % 10000)
        row[f"{metric}_patient_boot_ci_low"] = low
        row[f"{metric}_patient_boot_ci_high"] = high
    return row


def add_cluster_bootstrap_intervals(
    row: Dict[str, float], pred_df: pd.DataFrame,
    n_boot: int, seed: int
) -> Dict[str, float]:
    if "hospitalid" not in pred_df.columns:
        return row
    for metric in [
        "roc_auc", "pr_auc", "brier",
        "calibration_intercept", "calibration_slope", "oe_ratio"
    ]:
        low, high = cluster_bootstrap_metric_ci(
            pred_df,
            y_col="outcome",
            p_col="predicted_probability",
            cluster_col="hospitalid",
            metric=metric,
            n_boot=n_boot,
            seed=seed + hash(metric) % 10000,
        )
        row[f"{metric}_hospital_boot_ci_low"] = low
        row[f"{metric}_hospital_boot_ci_high"] = high
    return row
