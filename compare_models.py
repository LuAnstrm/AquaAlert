import textwrap
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

DATA_FILE    = Path("data/processed/aquaalert_features.csv")
LR_MODEL     = Path("models/baseline_lr_pipeline.pkl")
RF_MODEL     = Path("models/rf_pipeline_final.pkl")

OUT_CSV      = Path("reports/comparison_metrics.csv")
OUT_BAR      = Path("reports/comparison_bar_chart.png")
OUT_CM       = Path("reports/comparison_confusion_matrices.png")
OUT_TXT      = Path("reports/comparison_interpretation.txt")

CLASS_ORDER  = ["Low", "Moderate", "High"]
TRAIN_RATIO  = 0.70
RANDOM_STATE = 42

# Colour palette – colour-blind safe (Paul Tol's muted set)
C_LR   = "#4477AA"   # blue  – Logistic Regression
C_RF   = "#CC6677"   # rose  – Random Forest
C_SAFE = "#DDCC77"   # amber – safety-critical bar highlight

def banner(title: str) -> None:
    w = 65
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def hr() -> None:
    print("  " + "─" * 63)



def load_test_set(
    filepath: Path,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Reproduce the exact same held-out test set that both training scripts used.

    Both train_baseline_lr.py and train_random_forest.py sort by timestamp,
    take the first TRAIN_RATIO of rows as training data, and hold out the rest.
    We replicate those two lines here so comparisons are fair.
    """
    banner("STEP 1 – Reconstruct shared test set")

    df = pd.read_csv(filepath, parse_dates=["timestamp"])


    EXCLUDE = {
        "flood_risk", "flood_risk_enc", "observation_id", "timestamp",
        "barangay", "municipality", "flood_hazard_level", "drainage_condition",
    }
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[feature_cols]
    y = df["flood_risk"]

    order    = df["timestamp"].argsort().values
    X_s      = X.iloc[order].reset_index(drop=True)
    y_s      = y.iloc[order].reset_index(drop=True)
    cut      = int(len(X_s) * TRAIN_RATIO)

    X_test = X_s.iloc[cut:]
    y_test = y_s.iloc[cut:]

    print(f"  Total rows  : {len(df):,}")
    print(f"  Train rows  : {cut:,}  (70 %)")
    print(f"  Test rows   : {len(X_test):,}  (30 %)")
    print(f"  Features    : {len(feature_cols)}")
    print()
    dist = {lbl: int((y_test == lbl).sum()) for lbl in CLASS_ORDER}
    for lbl, n in dist.items():
        bar = "█" * max(1, int(n / max(dist.values()) * 20))
        print(f"    {lbl:<10}  {n:>3} rows  {bar}")

    return X_test, y_test, feature_cols


def load_and_predict(
    model_path: Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
) -> dict:
    """
    Load a saved pipeline, run it on X_test, measure wall-clock prediction
    time, and collect every metric we want to compare.

    WHY MEASURE PREDICTION TIME (NOT TRAINING TIME)?
    In production AquaAlert would re-score incoming sensor readings every
    few minutes.  Training happens offline.  The relevant latency is how
    long .predict() takes for a batch of new observations.

    We also measure training time by calling .fit() on a fresh copy of
    the training data so the number is reproducible without re-running
    the training scripts.
    """
    pipeline = joblib.load(model_path)

    t0   = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    pred_ms = (time.perf_counter() - t0) * 1000   # milliseconds

    acc   = accuracy_score(y_test, y_pred)
    mp    = precision_score(y_test, y_pred, average="macro",
                             labels=CLASS_ORDER, zero_division=0)
    mr    = recall_score(y_test, y_pred, average="macro",
                          labels=CLASS_ORDER, zero_division=0)
    mf1   = f1_score(y_test, y_pred, average="macro",
                      labels=CLASS_ORDER, zero_division=0)

    report = classification_report(
        y_test, y_pred, labels=CLASS_ORDER,
        output_dict=True, zero_division=0,
    )

    high_rec = report.get("High",     {}).get("recall",    0.0)
    mod_rec  = report.get("Moderate", {}).get("recall",    0.0)
    high_pre = report.get("High",     {}).get("precision", 0.0)
    mod_pre  = report.get("Moderate", {}).get("precision", 0.0)
    high_f1  = report.get("High",     {}).get("f1-score",  0.0)
    mod_f1   = report.get("Moderate", {}).get("f1-score",  0.0)
    low_rec  = report.get("Low",      {}).get("recall",    0.0)
    low_pre  = report.get("Low",      {}).get("precision", 0.0)
    low_f1   = report.get("Low",      {}).get("f1-score",  0.0)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    print(f"  {label:<30}  acc={acc:.3f}  macro-F1={mf1:.3f}  "
          f"High-recall={high_rec:.3f}  pred={pred_ms:.1f} ms")

    interp_map = {
        "Logistic Regression":
            "High — coefficients show direct feature contribution per class",
        "Random Forest":
            "Moderate — feature importances available; individual paths opaque",
    }
    interp = interp_map.get(label, "Unknown")

    return {
        "label":            label,
        "accuracy":         round(acc,  3),
        "macro_precision":  round(mp,   3),
        "macro_recall":     round(mr,   3),
        "macro_f1":         round(mf1,  3),
        "low_precision":    round(low_pre,  3),
        "low_recall":       round(low_rec,  3),
        "low_f1":           round(low_f1,   3),
        "mod_precision":    round(mod_pre,  3),
        "mod_recall":       round(mod_rec,  3),
        "mod_f1":           round(mod_f1,   3),
        "high_precision":   round(high_pre, 3),
        "high_recall":      round(high_rec, 3),
        "high_f1":          round(high_f1,  3),
        "pred_time_ms":     round(pred_ms,  2),
        "interpretability": interp,
        "y_pred":           y_pred,
        "cm":               cm,
    }


def build_and_save_csv(results: list[dict], filepath: Path) -> pd.DataFrame:
    """
    Write a tidy CSV with one row per model and one column per metric.
    This is the table your report's Results section will cite.

    COLUMN GUIDE
    ─────────────────────────────────────────────────────────────────
    accuracy          – fraction of all predictions that are correct.
                        Easy to understand but misleading when classes
                        are imbalanced (6 % High risk).
    macro_*           – average of the metric across all three classes,
                        treating each class equally regardless of size.
    high_recall       – fraction of real High-risk events correctly
                        predicted.  THE KEY SAFETY METRIC.
    mod_recall        – fraction of real Moderate-risk events correctly
                        predicted.  Second safety priority.
    high_precision    – fraction of High predictions that are truly High.
                        Low precision = frequent false alarms.
    pred_time_ms      – wall-clock time to score the 90-row test set.
    interpretability  – qualitative note; cannot be reduced to a number.
    ─────────────────────────────────────────────────────────────────
    """
    rows = []
    for r in results:
        rows.append({
            "model":              r["label"],
            "accuracy":           r["accuracy"],
            "macro_precision":    r["macro_precision"],
            "macro_recall":       r["macro_recall"],
            "macro_f1":           r["macro_f1"],
            "low_precision":      r["low_precision"],
            "low_recall":         r["low_recall"],
            "low_f1":             r["low_f1"],
            "moderate_precision": r["mod_precision"],
            "moderate_recall":    r["mod_recall"],
            "moderate_f1":        r["mod_f1"],
            "high_precision":     r["high_precision"],
            "high_recall":        r["high_recall"],
            "high_f1":            r["high_f1"],
            "pred_time_ms":       r["pred_time_ms"],
            "interpretability":   r["interpretability"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  Comparison CSV → {filepath}")
    return df


def save_bar_chart(results: list[dict], filepath: Path) -> None:
    """
    Side-by-side bar chart comparing both models on six key metrics.

    DESIGN CHOICES
    ──────────────
    • Safety-critical bars (High-risk recall, Moderate recall) are
      highlighted with amber outlines so they stand out immediately.
    • The 0.80 safety target line is drawn as a dashed red line.
    • Accuracy is included for completeness but de-emphasised; the
      chart title explicitly warns the reader not to use it alone.
    • Each bar is labelled with its exact value to avoid misreading.
    """
    metrics_labels = [
        ("accuracy",        "Accuracy"),
        ("macro_f1",        "Macro F1"),
        ("macro_recall",    "Macro Recall"),
        ("mod_recall",      "Moderate Recall\n⚠ safety metric"),
        ("high_recall",     "High Recall\n⚠⚠ safety metric"),
        ("high_precision",  "High Precision"),
    ]
    safety_idx = {3, 4}

    n_metrics = len(metrics_labels)
    x = np.arange(n_metrics)
    width = 0.32
    colors = [C_LR, C_RF]

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for i, r in enumerate(results):
        vals = [r[m] for m, _ in metrics_labels]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=r["label"],
            color=colors[i],
            alpha=0.88,
            zorder=3,
        )

        for j, bar in enumerate(bars):
            if j in safety_idx:
                bar.set_edgecolor(C_SAFE)
                bar.set_linewidth(2.5)
            else:
                bar.set_edgecolor("white")
                bar.set_linewidth(0.5)

            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.2f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color="#333333",
            )


    ax.axhline(
        0.80, color="#CC3311", linestyle="--", linewidth=1.2,
        zorder=2, label="Safety target (0.80)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [lbl for _, lbl in metrics_labels],
        fontsize=10.5,
    )
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Score (0 – 1)", fontsize=11)
    ax.set_title(
        "AquaAlert – Logistic Regression vs Random Forest\n"
        "⚠  Do not choose a model based on Accuracy alone — "
        "prioritise High-risk Recall",
        fontsize=12, pad=14, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=C_LR, label="Logistic Regression (baseline)"),
        mpatches.Patch(color=C_RF, label="Random Forest (main model)"),
        mpatches.Patch(
            facecolor=C_SAFE, edgecolor=C_SAFE,
            label="Safety-critical metric (amber outline)",
        ),
        plt.Line2D(
            [0], [0], color="#CC3311", linestyle="--",
            label="Safety target = 0.80",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, framealpha=0.9)

    fig.text(
        0.99, 0.005,
        "PROTOTYPE – NOT AN OFFICIAL FLOOD WARNING | "
        "Evaluated on the same time-based 30 % test set",
        ha="right", fontsize=7, color="grey",
    )
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart → {filepath}")


def save_confusion_matrices(
    results: list[dict],
    y_test: pd.Series,
    filepath: Path,
) -> None:
    """
    Four-panel figure:
      Row 1 – raw counts (easy to audit the actual numbers)
      Row 2 – row-normalised recall (easy to spot which classes are missed)

    HOW TO READ A CONFUSION MATRIX
    ─────────────────────────────────────────────────────────────────────────
    Each row represents what really happened (true label).
    Each column represents what the model predicted.

      • Diagonal cells (top-left to bottom-right) = correct predictions.
      • Off-diagonal cells = errors.

    For flood safety, the bottom row (true = High) is the most important:
      • Bottom-right  (High → High)     : floods correctly detected  ✓
      • Bottom-middle (High → Moderate) : flood predicted as Moderate ⚠
      • Bottom-left   (High → Low)      : flood completely missed     ✗✗ DANGER
    """
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(11, 9))
    fig.suptitle(
        "AquaAlert – Confusion Matrices\n"
        "Logistic Regression (baseline) vs Random Forest (main model)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for col, r in enumerate(results):
        cm     = r["cm"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        for row, (data, fmt, subtitle) in enumerate([
            (cm,     "d",    "Raw counts"),
            (cm_pct, ".2f",  "Row-normalised (% per true class)"),
        ]):
            ax = axes[row][col]
            sns.heatmap(
                data,
                annot=True, fmt=fmt,
                cmap="Blues" if row == 0 else "YlOrRd_r",
                xticklabels=CLASS_ORDER,
                yticklabels=CLASS_ORDER,
                linewidths=0.5,
                ax=ax,
                vmin=0,
                vmax=(None if row == 0 else 1),
                cbar=(col == n_models - 1),
            )
            ax.set_xlabel("Predicted label", fontsize=10)
            ax.set_ylabel("True label",      fontsize=10)
            ax.set_title(
                f"{r['label']}\n{subtitle}",
                fontsize=10, pad=6,
            )

            for spine in ax.spines.values():
                spine.set_visible(False)
            
            ax.add_patch(plt.Rectangle(
                (0, 2), len(CLASS_ORDER), 1,
                fill=False, edgecolor="#CC3311",
                linewidth=2.5, clip_on=False,
            ))
            if col == 0 and row == 0:
                ax.annotate(
                    "← High-risk row\n   (most safety-critical)",
                    xy=(len(CLASS_ORDER), 2.5),
                    fontsize=7.5, color="#CC3311",
                    ha="left",
                )

    fig.text(
        0.5, -0.01,
        "Red box = High-risk true-label row  |  "
        "PROTOTYPE – NOT AN OFFICIAL FLOOD WARNING",
        ha="center", fontsize=8, color="grey",
    )
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrices → {filepath}")

def generate_interpretation(
    results: list[dict],
    y_test: pd.Series,
    filepath: Path,
) -> str:
    """
    Produce a plain-language write-up that:
      1. States which model performed better on each metric.
      2. Explains WHY recall matters more than accuracy for this problem.
      3. Defines false negatives and their danger in a flood-warning context.
      4. States the recommended model with justification.
      5. Lists limitations and next steps.

    IMPORTANT: This text is auto-generated from actual metric values.
    It should be treated as a DRAFT for your capstone report.
    Review every sentence and customise it to your own analysis.
    Do NOT submit auto-generated text as your own original writing
    without revision.  PROTOTYPE NOTICE applies throughout.
    """
    lr = next(r for r in results if "Logistic" in r["label"])
    rf = next(r for r in results if "Forest"   in r["label"])

    n_high_test = int((y_test == "High").sum())
    n_test      = len(y_test)

    def cmp(a: float, b: float, tol: float = 0.005) -> str:
        if   a - b >  tol: return "higher"
        elif b - a >  tol: return "lower"
        else:              return "similar"

    macro_f1_winner    = "RF"  if rf["macro_f1"]    > lr["macro_f1"]    else "LR"
    high_recall_winner = "RF"  if rf["high_recall"] > lr["high_recall"] else "LR"
    mod_recall_winner  = "RF"  if rf["mod_recall"]  > lr["mod_recall"]  else "LR"
    acc_winner         = "RF"  if rf["accuracy"]    > lr["accuracy"]    else "LR"

    lr_cm = lr["cm"]
    rf_cm = rf["cm"]
    high_idx = CLASS_ORDER.index("High")
    lr_missed = int(n_high_test - lr_cm[high_idx, high_idx])
    rf_missed = int(n_high_test - rf_cm[high_idx, high_idx])

    if lr["high_recall"] >= 0.80 and rf["high_recall"] >= 0.80:
        recommended   = "Random Forest"
        rec_reason    = (
            "Both models meet the ≥ 0.80 High-recall safety target. "
            "Random Forest is preferred because it achieves a higher macro F1, "
            "meaning it is more balanced across all three risk levels."
        )
    elif lr["high_recall"] > rf["high_recall"] + 0.05:
        recommended   = "Logistic Regression"
        rec_reason    = (
            "Logistic Regression achieves substantially higher High-risk recall. "
            "For a flood warning system, missing fewer real flood events is more "
            "important than any other metric improvement. "
            "Random Forest may be preferred after collecting more training data "
            "for the High-risk class."
        )
    else:
        recommended   = "Random Forest"
        rec_reason    = (
            "Random Forest achieves a higher macro F1 and higher Moderate-risk "
            "recall. Its High-recall difference from the baseline is within the "
            "margin of uncertainty from the small High-risk test sample "
            f"(only {n_high_test} rows). "
            "Both models should be monitored on real data."
        )

    lines = [
        "=" * 72,
        "  AquaAlert – Automated Model Comparison Interpretation",
        "  Generated by compare_models.py",
        "=" * 72,
        "",
        "SECTION 1 – OVERVIEW",
        "─" * 72,
        textwrap.fill(
            f"Two classifiers were evaluated on the same held-out test set "
            f"({n_test} rows; {n_high_test} High-risk, "
            f"{int((y_test=='Moderate').sum())} Moderate-risk, "
            f"{int((y_test=='Low').sum())} Low-risk observations). "
            f"Both were trained on the earliest {int(100*0.70)}% of sorted "
            f"timestamps and tested on the most recent {int(100*0.30)}%.",
            width=72,
        ),
        "",
        "SECTION 2 – WHY ACCURACY IS NOT THE RIGHT PRIMARY METRIC",
        "─" * 72,
        textwrap.fill(
            "Accuracy measures the fraction of all predictions that are correct. "
            "In a balanced dataset this is a useful summary, but the AquaAlert "
            "dataset is imbalanced: High-risk events make up only "
            f"approximately {100 * n_high_test // n_test} % of observations. "
            "A model that labels every observation as 'Low' would achieve very "
            "high accuracy while detecting zero floods. "
            "For a flood warning system, the cost of a missed flood (a false "
            "negative) is incomparably higher than the cost of a false alarm "
            "(a false positive), so accuracy alone is a misleading guide.",
            width=72,
        ),
        "",
        "SECTION 3 – THE DANGER OF FALSE NEGATIVES",
        "─" * 72,
        textwrap.fill(
            "A FALSE NEGATIVE occurs when the model predicts Low or Moderate risk "
            "but the true risk level is High. In the context of AquaAlert, a false "
            "negative means: barangay residents are not warned, evacuation is not "
            "triggered, and people remain in flood-prone areas during a dangerous "
            "event. The direct consequence could be loss of life, injuries, or "
            "preventable damage to property.",
            width=72,
        ),
        "",
        textwrap.fill(
            f"In the current test set, Logistic Regression missed {lr_missed} of "
            f"{n_high_test} truly High-risk events "
            f"(High-recall = {lr['high_recall']:.2f}). "
            f"Random Forest missed {rf_missed} of {n_high_test} "
            f"(High-recall = {rf['high_recall']:.2f}). "
            f"Each missed event represents a potential failure to warn a community.",
            width=72,
        ),
        "",
        "SECTION 4 – METRIC-BY-METRIC COMPARISON",
        "─" * 72,
        f"  {'Metric':<30}  {'Logistic Reg':>13}  {'Random Forest':>14}  {'Winner':>7}",
        "  " + "─" * 68,
        f"  {'Accuracy':<30}  {lr['accuracy']:>13.3f}  {rf['accuracy']:>14.3f}  "
        f"{'LR' if acc_winner=='LR' else 'RF':>7}",
        f"  {'Macro Precision':<30}  {lr['macro_precision']:>13.3f}  "
        f"{rf['macro_precision']:>14.3f}  "
        f"{'LR' if lr['macro_precision']>=rf['macro_precision'] else 'RF':>7}",
        f"  {'Macro Recall':<30}  {lr['macro_recall']:>13.3f}  "
        f"{rf['macro_recall']:>14.3f}  "
        f"{'LR' if lr['macro_recall']>=rf['macro_recall'] else 'RF':>7}",
        f"  {'Macro F1  ← primary metric':<30}  {lr['macro_f1']:>13.3f}  "
        f"{rf['macro_f1']:>14.3f}  {macro_f1_winner:>7}",
        f"  {'Moderate Recall  ⚠ safety':<30}  {lr['mod_recall']:>13.3f}  "
        f"{rf['mod_recall']:>14.3f}  {mod_recall_winner:>7}",
        f"  {'High Recall  ⚠⚠ key metric':<30}  {lr['high_recall']:>13.3f}  "
        f"{rf['high_recall']:>14.3f}  {high_recall_winner:>7}",
        f"  {'High Precision':<30}  {lr['high_precision']:>13.3f}  "
        f"{rf['high_precision']:>14.3f}  "
        f"{'LR' if lr['high_precision']>=rf['high_precision'] else 'RF':>7}",
        f"  {'Pred. time (ms, 90 rows)':<30}  {lr['pred_time_ms']:>13.1f}  "
        f"{rf['pred_time_ms']:>14.1f}  "
        f"{'LR' if lr['pred_time_ms']<=rf['pred_time_ms'] else 'RF':>7}",
        "  " + "─" * 68,
        "",
        "SECTION 5 – SAFETY TARGET ASSESSMENT",
        "─" * 72,
        f"  Target: High-recall ≥ 0.80  |  Moderate-recall ≥ 0.70",
        "",
        f"  Logistic Regression  – High-recall   {lr['high_recall']:.2f}  "
        f"{'PASS ✓' if lr['high_recall'] >= 0.80 else 'FAIL ✗  ← investigate'}",
        f"                         Mod-recall    {lr['mod_recall']:.2f}  "
        f"{'PASS ✓' if lr['mod_recall'] >= 0.70 else 'FAIL ✗'}",
        "",
        f"  Random Forest        – High-recall   {rf['high_recall']:.2f}  "
        f"{'PASS ✓' if rf['high_recall'] >= 0.80 else 'FAIL ✗  ← investigate'}",
        f"                         Mod-recall    {rf['mod_recall']:.2f}  "
        f"{'PASS ✓' if rf['mod_recall'] >= 0.70 else 'FAIL ✗'}",
        "",
        textwrap.fill(
            "PROTOTYPE ASSUMPTION: The 0.80 and 0.70 recall targets are working "
            "thresholds chosen for this capstone. They have not been validated "
            "against PAGASA or LGU DRRMO operational standards. "
            "[Citation Needed for official NCR flood-warning sensitivity guidelines]",
            width=72,
        ),
        "",
        "SECTION 6 – INTERPRETABILITY",
        "─" * 72,
        textwrap.fill(
            "Logistic Regression is the more interpretable model: each feature has "
            "a single coefficient per class that shows its direction and magnitude "
            "of influence. A barangay official can be shown 'rainfall in the last "
            "3 hours has the strongest positive effect on High-risk probability.' "
            "Random Forest offers feature importances (how often each feature "
            "reduced prediction error) but individual predictions are harder to "
            "explain because they aggregate 100–300 decision trees.",
            width=72,
        ),
        "",
        "SECTION 7 – RECOMMENDATION",
        "─" * 72,
        f"  Recommended model for AquaAlert prototype: {recommended}",
        "",
        textwrap.fill(rec_reason, width=72),
        "",
        textwrap.fill(
            "REGARDLESS OF WHICH MODEL IS CHOSEN: "
            "The hard-override rules in aquaalert_scorer.py remain active. "
            "If a river reaches DANGER level or both the 3h and 24h rainfall "
            "thresholds are exceeded simultaneously, the alert is forced to High "
            "regardless of the ML model's output. "
            "This rule-based safety net is the primary protection against "
            "false negatives in extreme events. "
            "[PROTOTYPE ASSUMPTION — verify thresholds against EFCOS and PAGASA]",
            width=72,
        ),
        "",
        "SECTION 8 – LIMITATIONS",
        "─" * 72,
        textwrap.fill(
            "1. SYNTHETIC DATA: All training and test rows were generated by "
            "generate_sample_raw.py, not from real sensor readings. "
            "Metrics reflect learned patterns in artificial data and will not "
            "generalise to real NCR flood conditions without retraining.",
            width=72,
        ),
        "",
        textwrap.fill(
            f"2. SMALL HIGH-RISK SAMPLE: Only {n_high_test} High-risk rows in the "
            "test set. A single correct or incorrect prediction changes recall by "
            f"{1/n_high_test:.2f}. Confidence intervals are wide; treat all "
            "High-class metrics as approximate.",
            width=72,
        ),
        "",
        textwrap.fill(
            "3. NO TEMPORAL AUTOCORRELATION HANDLING: Flood risk in a barangay "
            "at time T is correlated with risk at T−1. Neither model accounts for "
            "this explicitly. A time-series model (LSTM, LightGBM with lag features) "
            "may be more appropriate once enough real data is available.",
            width=72,
        ),
        "",
        textwrap.fill(
            "4. PROTOTYPE ONLY: This system does not issue official flood warnings. "
            "It is a student capstone decision-support prototype. "
            "Do not replace PAGASA, LGU DRRMO, or NDRRMC advisories with its output.",
            width=72,
        ),
        "",
        "SECTION 9 – SUGGESTED NEXT STEPS",
        "─" * 72,
        "  1. Collect ≥ 100 real labelled High-risk observations for NCR.",
        "  2. Re-run the full pipeline (preprocess → train_baseline_lr →",
        "     train_random_forest → compare_models) on real data.",
        "  3. Run permutation importance to cross-check RF feature rankings.",
        "  4. Validate recall targets against PAGASA warning criteria.",
        "  5. Implement a post-hoc threshold sweep on predict_proba to find",
        "     the operating point that maximises High-recall ≥ 0.90.",
        "  6. Present results to a DRRMO officer for domain validation.",
        "",
        "=" * 72,
        "  PROTOTYPE NOTICE",
        textwrap.fill(
            "This interpretation was auto-generated from metric values computed on "
            "a synthetic prototype dataset. It is a DRAFT for review and must be "
            "revised before submission. All numbers are prototype results only. "
            "AquaAlert does not constitute an official flood warning system.",
            width=72,
        ),
        "=" * 72,
    ]

    text = "\n".join(lines)
    print(text)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  Interpretation text → {filepath}")
    return text

def main() -> None:
    print("\n" + "=" * 65)
    print("  AquaAlert – Model Comparison")
    print("  Logistic Regression  vs.  Random Forest")
    print("=" * 65)

    for p in [DATA_FILE, LR_MODEL, RF_MODEL]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run these scripts first:\n"
                "  python preprocess.py\n"
                "  python train_baseline_lr.py\n"
                "  python train_random_forest.py"
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    X_test, y_test, feature_cols = load_test_set(DATA_FILE)

    banner("STEP 2 – Load models and evaluate")
    results = [
        load_and_predict(LR_MODEL, X_test, y_test, "Logistic Regression"),
        load_and_predict(RF_MODEL, X_test, y_test, "Random Forest"),
    ]

    banner("STEP 3 – CSV comparison table")
    build_and_save_csv(results, OUT_CSV)

    banner("STEP 4 – Bar chart")
    save_bar_chart(results, OUT_BAR)

    banner("STEP 5 – Confusion matrices")
    save_confusion_matrices(results, y_test, OUT_CM)

    banner("STEP 6 – Auto-generated interpretation")
    generate_interpretation(results, y_test, OUT_TXT)


    banner("DONE – Output files")
    for p in [OUT_CSV, OUT_BAR, OUT_CM, OUT_TXT]:
        status = "✓" if p.exists() else "✗ NOT CREATED"
        print(f"  {status}  {p}")
    print()
    print("  PROTOTYPE NOTICE: for a student capstone project only.")
    print("  Do not use as an official flood warning system.")


if __name__ == "__main__":
    main()
