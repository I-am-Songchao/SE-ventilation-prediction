# -*- coding: utf-8 -*-
"""
07_figures_tables.py
===================================
Final revised reproducible figure-generation script for the SE-MV manuscript.

Compared with the previous version, this revision implements the manuscript-
matched figure structure recommended after side-by-side comparison with the
current draft figures:
    - Figure 1 restores the temporal-validation split and the study-design note.
    - Figure 2 keeps the clearer sorted coefficient plot.
    - Figure 3 uses publication-style model labels and panel titles.
    - Figure 4 adds panel titles and bin-level sample-size annotations.
    - Figure 5 is narrowed back to the main 7-row validation/sensitivity display.
    - Supplementary Figure S1 adds the 80% threshold label and the number of
      hospitals with 0% early GCS motor documentation coverage.

Outputs:
    Figure1_Cohort_Flow.png/.pdf
    Figure2_Final_Ridge8_Standardized_Coefficients.png/.pdf
    Figure3_ROC_Temporal_and_External_Validation.png/.pdf
    Figure4_Calibration_Temporal_and_External_Validation.png/.pdf
    Figure5_AUROC_Forest_Primary_and_Key_Sensitivity_Analyses.png/.pdf
    Supplementary_Figure_S1_Hospital_GCS_Coverage.png/.pdf
    figure_generation_log.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, roc_auc_score

DPI = 600
MODEL_FINAL = "Ridge8_GCS"
MODEL_REDUCED = "Ridge7_harmonized"

FILES = {
    "mimic_landmark_flow": ["16_landmark形成流程.csv"],
    "mimic_final_cohort": ["17_排除缺氧心搏骤停后主队列汇总.csv"],
    "eicu_stage1_flow": ["59_eICU外部验证队列流程.csv"],
    "eicu_final_flow": ["93_eICU最终外部验证队列流程.csv"],
    "mimic_internal": ["94_MIMIC七八变量模型内部验证汇总.csv"],
    "final_formula": ["98_最终模型公式与预处理参数.csv"],
    "eicu_external": ["99_eICU原始外部验证性能汇总.csv"],
    "mimic_temporal": ["106_MIMIC七八变量时间验证汇总.csv"],
    "eicu_hospital_restrict": ["111_eICU按医院GCS覆盖限制验证.csv"],
    "patient_predictions": ["112_第六阶段患者级预测.csv"],
    "hospital_gcs_fallback": ["77_eICU医院层面GCS覆盖率.csv"],
}

FEATURE_LABELS_EN = {
    "age": "Age",
    "sex": "Female sex",
    "gcs_motor_landmark": "GCS motor score at 6 h",
    "map_min": "Minimum MAP",
    "spo2_min": "Minimum SpO$_2$",
    "fio2_max": "Maximum FiO$_2$",
    "peep_max": "Maximum PEEP",
    "bun_latest": "Latest BUN",
}

FEATURE_LABELS_ZH = {
    "age": "年龄",
    "sex": "女性",
    "gcs_motor_landmark": "6 h时GCS运动评分",
    "map_min": "最低MAP",
    "spo2_min": "最低SpO$_2$",
    "fio2_max": "最高FiO$_2$",
    "peep_max": "最高PEEP",
    "bun_latest": "最近一次BUN",
}

SENS_LABELS_EN = {
    "MIMIC repeated 5×5 CV": "MIMIC repeated 5×5 CV",
    "MIMIC temporal validation": "MIMIC temporal validation",
    "eICU primary GCS-observed": "eICU primary, GCS observed",
    "eICU confirmed invasive only": "eICU confirmed invasive only",
    "eICU landmark SE": "eICU SE documented by 6 h",
    "eICU all ventilation evidence": "eICU all ventilation evidence",
    "eICU anytime SE": "eICU SE documented at any time",
}

SENS_LABELS_ZH = {
    "MIMIC repeated 5×5 CV": "MIMIC重复5×5折交叉验证",
    "MIMIC temporal validation": "MIMIC时间验证",
    "eICU primary GCS-observed": "eICU主要队列（GCS有记录）",
    "eICU confirmed invasive only": "eICU仅确认有创通气",
    "eICU landmark SE": "eICU 6 h前已记录SE",
    "eICU all ventilation evidence": "eICU全部通气证据",
    "eICU anytime SE": "eICU任意时点记录SE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final revised manuscript figures for the SE-MV study.")
    parser.add_argument("--input-dir", action="append", dest="input_dirs", default=None,
                        help="Directory containing result CSV files. Repeat for multiple folders.")
    parser.add_argument("--output-dir", default="SE_MV_Final_Figures_v2",
                        help="Output directory for PNG/PDF figures and generation log.")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def configure_matplotlib(language: str) -> None:
    if language == "zh":
        plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei", "Songti SC", "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False


def discover_files(input_dirs: Sequence[Path]) -> Dict[str, Path]:
    basename_index: Dict[str, List[Path]] = {}
    for root in input_dirs:
        if not root.exists():
            print(f"[WARN] Input directory does not exist: {root}")
            continue
        for p in root.rglob("*.csv"):
            basename_index.setdefault(p.name, []).append(p)
    found: Dict[str, Path] = {}
    for key, candidate_names in FILES.items():
        matches: List[Path] = []
        for name in candidate_names:
            matches.extend(basename_index.get(name, []))
        if matches:
            matches = sorted(set(matches), key=lambda p: (len(str(p)), str(p)))
            found[key] = matches[0]
            if len(matches) > 1:
                print(f"[INFO] Multiple matches for {key}; using: {matches[0]}")
        else:
            print(f"[WARN] Missing input for {key}: {candidate_names}")
    return found


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {png.name}")
    print(f"[OK] {pdf.name}")


def require(found: Dict[str, Path], keys: Iterable[str], figure_name: str) -> bool:
    missing = [k for k in keys if k not in found]
    if missing:
        print(f"[SKIP] {figure_name}: missing {', '.join(missing)}")
        return False
    return True


def first_value(df: pd.DataFrame, col: str, default=np.nan):
    if col not in df.columns or len(df) == 0:
        return default
    return df.iloc[0][col]


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def safe_int(x) -> int:
    if pd.isna(x):
        return 0
    return int(round(float(x)))


# -----------------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------------
def draw_vertical_flow(ax: plt.Axes, labels: Sequence[str], panel: str, x0: float = 0.08, box_w: float = 0.84) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, panel)

    n = len(labels)
    ys = np.linspace(0.92, 0.10, n)
    box_h = min(0.11, 0.74 / max(n, 1))

    for i, (y, label) in enumerate(zip(ys, labels)):
        box = FancyBboxPatch((x0, y - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.012,rounding_size=0.015",
                             fill=False, linewidth=1.0)
        ax.add_patch(box)
        ax.text(x0 + box_w / 2, y, label, ha="center", va="center", fontsize=8.0, wrap=True)
        if i < n - 1:
            y_next = ys[i + 1]
            arrow = FancyArrowPatch((x0 + box_w / 2, y - box_h / 2 - 0.008),
                                    (x0 + box_w / 2, y_next + box_h / 2 + 0.008),
                                    arrowstyle="-|>", mutation_scale=9, linewidth=0.9)
            ax.add_patch(arrow)


def add_footer_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=8)


def figure1_cohort_flow(found: Dict[str, Path], output_dir: Path, dpi: int, language: str) -> None:
    keys = ["mimic_landmark_flow", "mimic_final_cohort", "eicu_stage1_flow", "eicu_final_flow", "mimic_temporal", "eicu_external"]
    if not require(found, keys, "Figure 1"):
        return

    mimic_flow = read_csv(found["mimic_landmark_flow"])
    mimic_final = read_csv(found["mimic_final_cohort"])
    eicu_stage1 = read_csv(found["eicu_stage1_flow"])
    eicu_final = read_csv(found["eicu_final_flow"])
    d106 = read_csv(found["mimic_temporal"])
    d99 = read_csv(found["eicu_external"])

    m_initial = safe_int(first_value(mimic_flow, "n_initial_minus6_to6h"))
    m_death = safe_int(first_value(mimic_flow, "n_death_before6h"))
    m_not_inv = safe_int(first_value(mimic_flow, "n_alive_but_not_invasive_at6h"))
    m_landmark = safe_int(first_value(mimic_flow, "n_final_landmark_cohort"))
    row_m = mimic_final[(mimic_final["cohort_type"] == "primary_no_postanoxic") & (mimic_final["stratum_type"] == "all")]
    m_final = safe_int(first_value(row_m, "n"))
    m_event = safe_int(first_value(row_m, "n_unfavorable72"))
    m_excluded_post = max(m_landmark - m_final, 0)

    r106 = d106[d106["model"] == MODEL_FINAL]
    m_train = safe_int(first_value(r106, "n_train"))
    m_train_event = safe_int(first_value(r106, "n_train_event"))
    m_temp = safe_int(first_value(r106, "n"))
    m_temp_event = safe_int(first_value(r106, "n_event"))

    e_dict = dict(zip(eicu_stage1["step"], eicu_stage1["n"]))
    e_adult_se = safe_int(e_dict.get("adult_SE_stays", np.nan))
    e_unique = safe_int(e_dict.get("one_SE_stay_per_patient", np.nan))
    e_no_post = safe_int(e_dict.get("one_SE_stay_no_postanoxic", np.nan))
    e_early_imv = safe_int(e_dict.get("early_IMV_minus6_plus6", np.nan))
    e_within24 = safe_int(e_dict.get("landmark_SE_within24h_no_postanoxic", np.nan))

    row_primary_any = eicu_final[eicu_final["dataset"] == "primary_within24h_confirmed_or_probable"]
    e_primary = safe_int(first_value(row_primary_any, "n"))
    # For events/hospitals in strict 8-variable external validation, use Ridge8 row from d99.
    row_strict = d99[(d99["dataset"] == "primary_within24h_confirmed_or_probable") & (d99["model"] == MODEL_FINAL)]
    e_gcs = safe_int(first_value(row_strict, "n"))
    e_event = safe_int(first_value(row_strict, "n_event"))
    e_hosp_gcs = safe_int(first_value(row_strict, "n_hospitals"))

    if language == "zh":
        mimic_labels = [
            f"SE相关住院中首次有创机械通气位于\n预设早期时间窗\nn={m_initial}",
            f"6 h预测时点前排除\n死亡 n={m_death}；6 h时已无有创通气 n={m_not_inv}\n合计排除 n={m_death + m_not_inv}",
            f"满足6 h预测时点条件\nn={m_landmark}",
            f"排除缺氧后/心搏骤停相关病例\nn={m_excluded_post}",
            f"最终MIMIC-IV主开发队列\nn={m_final}；结局事件={m_event}",
            f"模型开发/重复内部验证\n重复5折交叉验证×5次\nn={m_final}；结局事件={m_event}",
            f"时间拆分\n2008–2016：n={m_train}，事件={m_train_event}\n2017–2022：n={m_temp}，事件={m_temp_event}",
        ]
        eicu_labels = [
            f"eICU成人SE相关ICU住院\nn={e_adult_se}",
            f"每位患者保留1次SE相关ICU住院\nn={e_unique}",
            f"排除缺氧后/心搏骤停相关病例\nn={max(e_unique - e_no_post, 0)}；剩余 n={e_no_post}",
            f"首次有创机械通气位于预设早期时间窗\nn={e_early_imv}",
            f"SE在24 h内记录且满足6 h预测时点\nn={e_within24}",
            f"确认或很可能为有创机械通气\nn={e_primary}",
            f"严格8变量外部验证（早期GCS运动评分有记录）\nn={e_gcs}；结局事件={e_event}；医院数={e_hosp_gcs}",
        ]
        footer = "预测时点：首次有创机械通气开始后6 h；主要结局：72 h时仍未脱离有创/气切机械通气，或6–72 h内死亡"
    else:
        mimic_labels = [
            f"SE-related admissions with first IMV in the\nprespecified early window\nn={m_initial}",
            f"Excluded before the 6-h prediction time\nDeath: n={m_death}; no longer invasively ventilated: n={m_not_inv}\nTotal excluded: n={m_death + m_not_inv}",
            f"Eligible at the 6-h prediction time\nn={m_landmark}",
            f"Excluded post-anoxic/cardiac-arrest-related cases\nn={m_excluded_post}",
            f"Final MIMIC-IV primary development cohort\nn={m_final}; events={m_event}",
            f"Model development/internal validation\nRepeated 5-fold CV × 5 repeats\nn={m_final}; events={m_event}",
            f"Temporal split\n2008–2016: n={m_train}, events={m_train_event}\n2017–2022: n={m_temp}, events={m_temp_event}",
        ]
        eicu_labels = [
            f"Adult eICU stays with SE-related diagnosis\nn={e_adult_se}",
            f"One SE-related ICU stay retained per patient\nn={e_unique}",
            f"Excluded post-anoxic/cardiac-arrest-related cases\nn={max(e_unique - e_no_post, 0)}; remaining n={e_no_post}",
            f"First IMV in the prespecified early window\nn={e_early_imv}",
            f"SE documented within 24 h and eligible at the 6-h landmark\nn={e_within24}",
            f"Confirmed or probable invasive ventilation\nn={e_primary}",
            f"Strict 8-variable external validation\nObserved early GCS motor score\nn={e_gcs}; events={e_event}; hospitals={e_hosp_gcs}",
        ]
        footer = "Prediction landmark: 6 h after first invasive ventilation start; primary outcome: continued invasive/tracheostomy ventilation at 72 h or death between 6 and 72 h"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 8.2))
    draw_vertical_flow(axes[0], mimic_labels, "A")
    draw_vertical_flow(axes[1], eicu_labels, "B")
    fig.subplots_adjust(wspace=0.18, bottom=0.09)
    add_footer_note(fig, footer)
    save_figure(fig, output_dir, "Figure1_Cohort_Flow", dpi)


# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------
def figure2_coefficients(found: Dict[str, Path], output_dir: Path, dpi: int, language: str) -> None:
    if not require(found, ["final_formula"], "Figure 2"):
        return
    df = read_csv(found["final_formula"])
    sub = df[(df["model"] == MODEL_FINAL) & (df["term"] != "intercept")].copy()
    sub = sub.dropna(subset=["standardized_coefficient"])
    if sub.empty:
        print("[SKIP] Figure 2: no Ridge8 standardized coefficients found")
        return
    label_map = FEATURE_LABELS_ZH if language == "zh" else FEATURE_LABELS_EN
    sub["label"] = sub["term"].map(label_map).fillna(sub["term"])
    sub["abscoef"] = sub["standardized_coefficient"].abs()
    sub = sub.sort_values("abscoef", ascending=True)

    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    ax.barh(sub["label"], sub["standardized_coefficient"])
    ax.axvline(0, linestyle="--", linewidth=0.9)
    ax.set_xlabel("标准化岭回归系数" if language == "zh" else "Standardized ridge coefficient")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for y, val in enumerate(sub["standardized_coefficient"].to_numpy()):
        offset = 0.018 if val >= 0 else -0.018
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, y, f"{val:.2f}", va="center", ha=ha, fontsize=8)

    lim = max(abs(sub["standardized_coefficient"].min()), abs(sub["standardized_coefficient"].max())) + 0.15
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    save_figure(fig, output_dir, "Figure2_Final_Ridge8_Standardized_Coefficients", dpi)


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------
def subset_predictions(df: pd.DataFrame, analysis: str, model: str) -> pd.DataFrame:
    sub = df[(df["analysis"] == analysis) & (df["model"] == model)].copy()
    sub = sub.dropna(subset=["outcome", "predicted_probability"])
    return sub


def plot_roc(ax: plt.Axes, datasets: Sequence[Tuple[pd.DataFrame, str]], title: Optional[str] = None) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9, label="Reference")
    for sub, label in datasets:
        if len(sub) == 0 or sub["outcome"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub["outcome"], sub["predicted_probability"])
        auc = roc_auc_score(sub["outcome"], sub["predicted_probability"])
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{label} (AUROC={auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    if title:
        ax.set_title(title, pad=6)
    handles, labels = ax.get_legend_handles_labels()
    if labels and labels[0] == "Reference":
        labels[0] = "Reference" if ax.get_xlabel() == "1 − Specificity" else labels[0]
    ax.legend(handles, labels, loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure3_roc(found: Dict[str, Path], output_dir: Path, dpi: int, language: str) -> None:
    if not require(found, ["patient_predictions"], "Figure 3"):
        return
    pred = read_csv(found["patient_predictions"])
    mim7 = subset_predictions(pred, "MIMIC_2008_2016_to_2017_2022", MODEL_REDUCED)
    mim8 = subset_predictions(pred, "MIMIC_2008_2016_to_2017_2022", MODEL_FINAL)
    ext7 = subset_predictions(pred, "primary_GCS_observed_same_subset", MODEL_REDUCED)
    ext8 = subset_predictions(pred, "primary_GCS_observed_same_subset", MODEL_FINAL)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    title_left = "MIMIC-IV temporal validation" if language == "en" else "MIMIC-IV时间验证"
    title_right = "eICU-CRD external validation" if language == "en" else "eICU-CRD外部验证"
    plot_roc(axes[0], [(mim7, "7-variable model"), (mim8, "8-variable model")], title_left)
    plot_roc(axes[1], [(ext7, "7-variable model"), (ext8, "8-variable model")], title_right)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")

    if language == "zh":
        for ax in axes:
            ax.set_xlabel("1 − 特异度")
            ax.set_ylabel("灵敏度")
            handles, labels = ax.get_legend_handles_labels()
            labels = ["参考线" if x == "Reference" else x for x in labels]
            ax.legend(handles, labels, loc="lower right", frameon=False)

    fig.tight_layout()
    save_figure(fig, output_dir, "Figure3_ROC_Temporal_and_External_Validation", dpi)


# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------
def calibration_bins(sub: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    x = sub[["outcome", "predicted_probability"]].dropna().copy()
    if len(x) == 0:
        return pd.DataFrame(columns=["mean_predicted", "observed", "n"])
    ranks = x["predicted_probability"].rank(method="first")
    q = min(n_bins, max(2, len(x) // 5))
    x["bin"] = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
    out = (
        x.groupby("bin", observed=True)
         .agg(mean_predicted=("predicted_probability", "mean"),
              observed=("outcome", "mean"),
              n=("outcome", "size"))
         .reset_index(drop=True)
    )
    return out


def plot_calibration(ax: plt.Axes, sub: pd.DataFrame, label: str, title: Optional[str] = None, language: str = "en") -> None:
    bins = calibration_bins(sub, n_bins=5)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9, label="Ideal")
    if not bins.empty:
        ax.plot(bins["mean_predicted"], bins["observed"], marker="o", linewidth=1.5, label=label)
        for _, r in bins.iterrows():
            txt = f"n={int(r['n'])}"
            ax.annotate(txt, (r["mean_predicted"], r["observed"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed event rate")
    if title:
        ax.set_title(title, pad=6)
    handles, labels = ax.get_legend_handles_labels()
    if language == "zh":
        labels = ["理想校准" if x == "Ideal" else x for x in labels]
    ax.legend(handles, labels, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure4_calibration(found: Dict[str, Path], output_dir: Path, dpi: int, language: str) -> None:
    if not require(found, ["patient_predictions"], "Figure 4"):
        return
    pred = read_csv(found["patient_predictions"])
    mim8 = subset_predictions(pred, "MIMIC_2008_2016_to_2017_2022", MODEL_FINAL)
    ext8 = subset_predictions(pred, "primary_GCS_observed_same_subset", MODEL_FINAL)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    title_left = "MIMIC-IV temporal validation" if language == "en" else "MIMIC-IV时间验证"
    title_right = "eICU-CRD external validation" if language == "en" else "eICU-CRD外部验证"
    plot_calibration(axes[0], mim8, "8-variable model" if language == "en" else "8变量模型", title_left, language)
    plot_calibration(axes[1], ext8, "8-variable model" if language == "en" else "8变量模型", title_right, language)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")

    if language == "zh":
        for ax in axes:
            ax.set_xlabel("预测风险")
            ax.set_ylabel("观察事件率")

    fig.tight_layout()
    save_figure(fig, output_dir, "Figure4_Calibration_Temporal_and_External_Validation", dpi)


# -----------------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------------
def add_forest_row(rows: List[dict], label: str, n: int, events: int, auc: float, low: float, high: float) -> None:
    rows.append({
        "label": label,
        "n": safe_int(n),
        "events": safe_int(events),
        "auc": float(auc),
        "low": float(low),
        "high": float(high),
    })


def figure5_auc_forest(found: Dict[str, Path], output_dir: Path, dpi: int, language: str) -> None:
    needed = ["mimic_internal", "mimic_temporal", "eicu_external"]
    if not require(found, needed, "Figure 5"):
        return
    rows: List[dict] = []

    d94 = read_csv(found["mimic_internal"])
    r = d94[d94["model"] == MODEL_FINAL]
    if not r.empty:
        rr = r.iloc[0]
        add_forest_row(rows, "MIMIC repeated 5×5 CV", rr.n, rr.n_event, rr.roc_auc, rr.roc_auc_ci_low, rr.roc_auc_ci_high)

    d106 = read_csv(found["mimic_temporal"])
    r = d106[d106["model"] == MODEL_FINAL]
    if not r.empty:
        rr = r.iloc[0]
        add_forest_row(rows, "MIMIC temporal validation", rr.n, rr.n_event, rr.roc_auc, rr.roc_auc_patient_boot_ci_low, rr.roc_auc_patient_boot_ci_high)

    d99 = read_csv(found["eicu_external"])
    mapping = [
        ("primary_within24h_confirmed_or_probable", "eICU primary GCS-observed"),
        ("sensitivity_within24h_confirmed_only", "eICU confirmed invasive only"),
        ("sensitivity_landmark_SE_confirmed_or_probable", "eICU landmark SE"),
        ("sensitivity_within24h_all_ventilation_evidence", "eICU all ventilation evidence"),
        ("sensitivity_anytime_SE_confirmed_or_probable", "eICU anytime SE"),
    ]
    for dataset, label in mapping:
        r = d99[(d99["dataset"] == dataset) & (d99["model"] == MODEL_FINAL)]
        if not r.empty:
            rr = r.iloc[0]
            add_forest_row(rows, label, rr.n, rr.n_event, rr.roc_auc,
                           rr.roc_auc_patient_boot_ci_low, rr.roc_auc_patient_boot_ci_high)

    if not rows:
        print("[SKIP] Figure 5: no forest rows assembled")
        return

    df = pd.DataFrame(rows)
    label_map = SENS_LABELS_ZH if language == "zh" else SENS_LABELS_EN
    df["display"] = df["label"].map(label_map).fillna(df["label"])
    df["display"] = df.apply(lambda x: f"{x['display']}  (n={x['n']}, events={x['events']})", axis=1)
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    xerr = np.vstack([df["auc"] - df["low"], df["high"] - df["auc"]])

    fig_h = max(4.6, 0.52 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    ax.errorbar(df["auc"], y, xerr=xerr, fmt="o", capsize=3, linewidth=1.1)
    ax.axvline(0.5, linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"])
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC (95% CI)" if language == "en" else "AUROC（95%置信区间）")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, (auc, low, high) in enumerate(zip(df["auc"], df["low"], df["high"])):
        ax.text(min(high + 0.015, 0.985), yi, f"{auc:.3f} ({low:.3f}–{high:.3f})", va="center", fontsize=7.5)

    fig.tight_layout()
    save_figure(fig, output_dir, "Figure5_AUROC_Forest_Primary_and_Key_Sensitivity_Analyses", dpi)


# -----------------------------------------------------------------------------
# Supplementary Figure S1
# -----------------------------------------------------------------------------
def primary_hospital_coverage_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    sub = pred[(pred["analysis"] == "primary_all_patients_MIMIC_median_imputation") & (pred["model"] == MODEL_FINAL)].copy()
    sub = sub.dropna(subset=["hospitalid"])
    if sub.empty:
        return pd.DataFrame()
    sub["gcs_observed"] = sub["gcs_motor_landmark"].notna().astype(int)
    out = (sub.groupby("hospitalid", as_index=False)
             .agg(n=("outcome", "size"), n_outcome=("outcome", "sum"), n_gcs=("gcs_observed", "sum")))
    out["coverage_pct"] = 100 * out["n_gcs"] / out["n"]
    return out


def figure_s1_hospital_gcs(found: Dict[str, Path], output_dir: Path, dpi: int, language: str) -> None:
    coverage = pd.DataFrame()
    if "patient_predictions" in found:
        pred = read_csv(found["patient_predictions"])
        coverage = primary_hospital_coverage_from_predictions(pred)
    if coverage.empty and "hospital_gcs_fallback" in found:
        d77 = read_csv(found["hospital_gcs_fallback"])
        sub = d77[d77["cohort_definition"].isin(["SE_within24h_no_postanoxic", "SE_by_landmark_no_postanoxic"])].copy()
        if not sub.empty:
            sub = sub.sort_values(["hospitalid"]).drop_duplicates("hospitalid", keep="last")
            coverage = pd.DataFrame({
                "hospitalid": sub["hospitalid"],
                "n": sub["n"],
                "n_outcome": sub["n_outcome"],
                "n_gcs": sub["n_combined_gcs"],
                "coverage_pct": sub["combined_coverage_pct"],
            })
    if coverage.empty:
        print("[SKIP] Figure S1: no hospital-level GCS coverage data available")
        return

    coverage = coverage.sort_values(["coverage_pct", "n"], ascending=[True, True]).reset_index(drop=True)
    coverage["hospital_order"] = np.arange(1, len(coverage) + 1)
    n_zero = int((coverage["coverage_pct"] == 0).sum())

    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    ax.bar(coverage["hospital_order"], coverage["coverage_pct"])
    ax.axhline(80, linestyle="--", linewidth=0.9)
    ax.text(len(coverage) * 0.73, 82.5, "80% threshold" if language == "en" else "80%阈值", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, len(coverage) + 1)
    if language == "zh":
        ax.set_xlabel("医院（按GCS运动评分记录覆盖率排序）")
        ax.set_ylabel("早期GCS运动评分记录覆盖率（%）")
        note = f"其中{n_zero}家医院覆盖率为0%"
    else:
        ax.set_xlabel("Hospital, ordered by GCS motor documentation coverage")
        ax.set_ylabel("Early GCS motor documentation coverage (%)")
        note = f"{n_zero} hospitals had 0% coverage"
    ax.text(0.01, 0.98, note, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir, "Supplementary_Figure_S1_Hospital_GCS_Coverage", dpi)


# -----------------------------------------------------------------------------
# Log and QC
# -----------------------------------------------------------------------------
def write_generation_log(found: Dict[str, Path], output_dir: Path, language: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "figure_generation_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("SE-MV final revised figure generation log\n")
        f.write("=======================================\n")
        f.write(f"Language: {language}\n")
        f.write(f"PNG dpi: {dpi}\n\n")
        f.write("Resolved input files:\n")
        for key in sorted(FILES):
            f.write(f"  {key}: {found.get(key, 'MISSING')}\n")
        f.write("\nImportant: patient-level predictions are local plotting inputs only and should not be publicly shared.\n")
    print(f"[OK] {log_path.name}")


def run_qc(found: Dict[str, Path]) -> None:
    if "patient_predictions" not in found:
        return
    pred = read_csv(found["patient_predictions"])
    checks = [
        ("MIMIC_2008_2016_to_2017_2022", MODEL_FINAL),
        ("primary_GCS_observed_same_subset", MODEL_FINAL),
    ]
    for analysis, model in checks:
        sub = subset_predictions(pred, analysis, model)
        if len(sub) == 0 or sub["outcome"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["outcome"], sub["predicted_probability"])
        print(f"[QC] {analysis} / {model}: patient-level AUROC={auc:.6f}")


def main() -> int:
    args = parse_args()
    input_dirs = [Path(p).expanduser().resolve() for p in (args.input_dirs or [os.getcwd()])]
    output_dir = Path(args.output_dir).expanduser().resolve()

    configure_matplotlib(args.language)
    found = discover_files(input_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nResolved inputs:")
    for key, path in sorted(found.items()):
        print(f"  {key}: {path}")
    print()

    run_qc(found)
    figure1_cohort_flow(found, output_dir, args.dpi, args.language)
    figure2_coefficients(found, output_dir, args.dpi, args.language)
    figure3_roc(found, output_dir, args.dpi, args.language)
    figure4_calibration(found, output_dir, args.dpi, args.language)
    figure5_auc_forest(found, output_dir, args.dpi, args.language)
    figure_s1_hospital_gcs(found, output_dir, args.dpi, args.language)
    write_generation_log(found, output_dir, args.language, args.dpi)

    print(f"\nDone. Figures saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
