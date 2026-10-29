"""
train_random_forest.py
======================
AquaAlert – Random Forest Flood-Risk Classifier  (Main Model)
--------------------------------------------------------------

PURPOSE OF THIS SCRIPT
-----------------------
This script trains the MAIN AquaAlert model: a Random Forest classifier
that predicts flood risk (Low / Moderate / High) for a daily city observation.

HOW IT FITS INTO THE PROJECT
-----------------------------
                                         ┌──────────────────────┐
   preprocess.py                         │  You are here        │
   (clean CSV)  ──►  train_baseline_lr.py (Logistic Regression) │
                     (baseline model)   └──────────────────────┘
                              │
                              │  compare macro F1 + High-recall
                              ▼
                     train_random_forest.py   ◄──  this script
                     (main model)

A BASELINE (Logistic Regression) was already trained first.
The numbers at the bottom of this script should be compared against
that baseline to decide whether the extra complexity of Random Forest
is justified.

WHAT IS A RANDOM FOREST? (plain language)
------------------------------------------
A single Decision Tree asks a series of yes/no questions
("Is daily rainfall > 30 mm?  Is river level above 2.5 m?")
and follows a path to a prediction.  It is easy to understand but
tends to memorise the training data (overfitting).

A RANDOM FOREST builds MANY trees (e.g. 300), each trained on a
random sample of the data and a random subset of features.
Then it takes a majority vote across all trees.  Because each tree
is slightly different, their errors cancel out and the forest is
much harder to overfit than a single tree.

Random Forests handle:
  • non-linear feature interactions (e.g. high rainfall AND saturated soil)
  • different feature scales without needing StandardScaler
  • missing-value indicators (the _missing flag columns)
  • imbalanced classes (via class_weight or SMOTE)

PROTOTYPE NOTICE
----------------
This is a student capstone prototype.
Its output is DECISION-SUPPORT ONLY and does NOT replace advisories
from PAGASA, LGU DRRMOs, or the NDRRMC.
All parameter choices are PROTOTYPE ASSUMPTIONS.

HOW TO RUN
----------
    python train_random_forest.py

INPUT  : data/processed/aquaalert_features.csv
OUTPUTS (all in models/ and reports/):
    models/rf_pipeline_final.pkl           – trained pipeline
    reports/rf_classification_report.csv   – per-class precision / recall / F1
    reports/rf_confusion_matrix.png        – confusion matrix heatmap
    reports/rf_feature_importance.png      – bar chart of top feature importances
    reports/rf_feature_importance.csv      – importances as a table
    reports/rf_cv_results.csv              – cross-validation grid-search scores
    reports/rf_vs_baseline_summary.txt     – comparison table vs. Logistic Regression
"""

import warnings
import textwrap
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler       

warnings.filterwarnings("ignore")

DATA_FILE   = Path("data/processed/aquaalert_features.csv")
MODEL_FILE  = Path("models/rf_pipeline_final.pkl")
CR_CSV      = Path("reports/rf_classification_report.csv")
CM_PNG      = Path("reports/rf_confusion_matrix.png")
FI_PNG      = Path("reports/rf_feature_importance.png")
FI_CSV      = Path("reports/rf_feature_importance.csv")
CV_CSV      = Path("reports/rf_cv_results.csv")
CMP_TXT     = Path("reports/rf_vs_baseline_summary.txt")
LR_CR_CSV   = Path("reports/lr_classification_report.csv")  

CLASS_ORDER = ["Low", "Moderate", "High"]

TRAIN_RATIO = 0.70

SMOTE_K_NEIGHBORS = 3

PARAM_GRID = {


    "classifier__n_estimators": [100, 200, 300],


    "classifier__max_depth": [None, 10, 20],


    "classifier__min_samples_leaf": [1, 2, 4],
}

CV_FOLDS = 5


CV_SCORING = "f1_macro"

RANDOM_STATE = 42

TOP_N_FEATURES = 15

for p in [MODEL_FILE, CR_CSV, CM_PNG, FI_PNG, CMP_TXT]:
    p.parent.mkdir(parents=True, exist_ok=True)


def banner(title: str) -> None:
    w = 65
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def load_and_validate(filepath: Path) -> pd.DataFrame:
    """
    Read the processed CSV and run basic sanity checks before training.

    WHY VALIDATE?
    We want to catch problems early (missing columns, wrong data types,
    unexpected NaN values) so we get a clear error message instead of a
    confusing crash inside the model training code.
    """
    banner("STEP 1 – Load and validate dataset")

    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            "Run preprocess.py first to generate the processed CSV."
        )

    df = pd.read_csv(filepath, parse_dates=["timestamp"])

    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {len(df.columns)}")
    print(f"  Date range   : {df['timestamp'].min().date()} → "
          f"{df['timestamp'].max().date()}")
    print()

    required = [
        "timestamp", "flood_risk",
        "rainfall_last_24h_mm", "rainfall_last_3d_mm",
        "river_water_level_m", "flood_hazard_level_enc",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns missing from dataset: {missing_cols}\n"
            "Re-run preprocess.py to regenerate the processed CSV."
        )
    print("  Required columns present ✓")

    bad_labels = set(df["flood_risk"].unique()) - set(CLASS_ORDER)
    if bad_labels:
        raise ValueError(
            f"Unexpected labels in flood_risk: {bad_labels}\n"
            f"Expected only: {CLASS_ORDER}"
        )
    print("  Target labels valid ✓")

    counts = df["flood_risk"].value_counts()
    total  = len(df)
    print("\n  Class distribution:")
    for lbl in CLASS_ORDER:
        n   = counts.get(lbl, 0)
        pct = 100 * n / total
        bar = "█" * int(pct / 3)
        print(f"    {lbl:<10}  {n:>4} rows  ({pct:.1f}%)  {bar}")

    n_high = counts.get("High", 0)
    if n_high < 20:
        print(
            f"\n  ⚠  WARNING: Only {n_high} High-risk rows in the full dataset.\n"
            "     SMOTE will generate synthetic rows to compensate, but the\n"
            "     model's High-class metrics will be unreliable until you\n"
            "     collect more real High-risk observations.\n"
            "     PROTOTYPE ASSUMPTION – document this limitation in your report."
        )

    numeric_check_cols = [
        "rainfall_last_24h_mm", "rainfall_last_7d_mm",
        "elevation_m", "soil_moisture_pct",
    ]
    nan_counts = {c: int(df[c].isna().sum())
                  for c in numeric_check_cols if c in df.columns}
    any_nan = {c: v for c, v in nan_counts.items() if v > 0}
    if any_nan:
        raise ValueError(
            f"Unexpected NaN values found after preprocessing: {any_nan}\n"
            "Re-run preprocess.py to fix missing values."
        )
    print("  No unexpected NaN values in key columns ✓")

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Define X (features) and y (target) by explicitly excluding non-feature columns.

    TARGET LEAKAGE GUARD
    --------------------
    flood_risk_enc is the numeric encoding of the target label.
    Including it in X would let the model cheat – it would "predict" flood
    risk by reading the answer directly.  Same principle applies to flood_risk.

    flood_occurrence is the source binary label the 3-class target is built
    from, so it is excluded for the same reason.

    The municipality text column is excluded too; the per-city aegis_* and
    elevation columns already carry its information as numbers.
    """
    banner("STEP 2 – Build feature matrix")

    EXCLUDE = {
        "flood_risk",
        "flood_risk_enc",
        "flood_occurrence",
        "observation_id",
        "timestamp",
        "municipality",
        "flood_hazard_level",
        "aegis_points",
    }

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols].copy()
    y = df["flood_risk"].copy()

    print(f"  Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"    • {col}")
    return X, y


def time_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sort by timestamp and split at the TRAIN_RATIO point.

    IMPORTANT: this produces the SAME test set as the Logistic Regression
    baseline (same ratio, same random_state = not used = deterministic sort).
    Using the same test set is required for a fair model comparison.

    SMOTE IS NOT APPLIED HERE.
    It is applied inside the cross-validation pipeline so synthetic rows
    are never created from test-fold data (that would be data leakage).
    """
    banner("STEP 3 – Time-based train / test split")

    order       = df["timestamp"].argsort().values
    X_s, y_s    = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    cut         = int(len(X_s) * TRAIN_RATIO)

    X_train, X_test = X_s.iloc[:cut], X_s.iloc[cut:]
    y_train, y_test = y_s.iloc[:cut], y_s.iloc[cut:]

    print(f"  Train : {len(X_train):,} rows  "
          f"(ends at row {cut - 1})")
    print(f"  Test  : {len(X_test):,} rows  "
          f"(rows {cut} → {len(X_s) - 1})")
    print()
    for name, yp in [("Train", y_train), ("Test", y_test)]:
        dist = {lbl: int((yp == lbl).sum()) for lbl in CLASS_ORDER}
        print(f"  {name} distribution: "
              + "  ".join(f"{k}={v}" for k, v in dist.items()))

    return X_train, X_test, y_train, y_test


def build_pipeline() -> ImbPipeline:
    """
    Assemble a three-step pipeline:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Step 1 – SMOTE (inside training folds only)                   │
    │    Creates synthetic High-risk rows so the classifier sees     │
    │    enough examples to learn what High risk looks like.         │
    │                                                                 │
    │  Step 2 – StandardScaler                                       │
    │    Random Forests do NOT require scaling, but we include it    │
    │    to keep this pipeline consistent with the baseline and to   │
    │    make the pipeline reusable if we swap in other classifiers. │
    │                                                                 │
    │  Step 3 – RandomForestClassifier                               │
    │    The main model.  Hyperparameters are tuned by GridSearchCV  │
    │    in the next step.  class_weight='balanced_subsample' is an  │
    │    additional safeguard: each bootstrap sample drawn per tree  │
    │    is re-weighted so the High class is never overlooked.       │
    └─────────────────────────────────────────────────────────────────┘

    WHY ImbPipeline INSTEAD OF THE STANDARD sklearn Pipeline?
    imbalanced-learn's Pipeline knows that SMOTE should only run during
    .fit() – not during .predict() or .transform().  The standard sklearn
    Pipeline would try to apply SMOTE on the test set too, which is wrong.
    """
    smote = SMOTE(
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()

    rf = RandomForestClassifier(

        class_weight="balanced_subsample",

        oob_score=True,

        n_jobs=-1,

        random_state=RANDOM_STATE,
    )

    pipeline = ImbPipeline(steps=[
        ("smote",      smote),
        ("scaler",     scaler),
        ("classifier", rf),
    ])
    return pipeline

def tune_hyperparameters(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[ImbPipeline, pd.DataFrame]:
    """
    Search the PARAM_GRID for the combination of hyperparameters that gives
    the highest macro F1 on a 5-fold stratified cross-validation of the
    TRAINING data only.

    HOW GridSearchCV WORKS (step by step)
    ──────────────────────────────────────
    1. It takes every combination of values in PARAM_GRID.
       (3 × 3 × 3 = 27 combinations here.)
    2. For each combination, it splits the training data into 5 folds.
    3. For each fold: train on 4 folds, validate on 1 fold, record macro F1.
       SMOTE runs only on the 4 training folds – never on the validation fold.
    4. Average the 5 fold scores.  The combination with the highest average
       macro F1 is chosen.
    5. The best pipeline is then re-trained on the FULL training set.

    IMPORTANT: the test set is never involved in this process.
    It is held completely separate until the final evaluation in Step 6.

    WHY IS THIS IMPORTANT FOR A FAIR EVALUATION?
    If we tuned hyperparameters using the test set, the test score would
    reflect how well we optimised for that specific test set rather than how
    well the model generalises to new, unseen data.
    """
    banner("STEP 5 – Hyperparameter tuning (GridSearchCV)")

    total_combos = 1
    for v in PARAM_GRID.values():
        total_combos *= len(v)
    print(f"  Searching {total_combos} hyperparameter combinations × {CV_FOLDS} folds "
          f"= {total_combos * CV_FOLDS} model fits …")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring=CV_SCORING,     
        refit=True,              
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)

    best_params = {
        k.replace("classifier__", ""): v
        for k, v in grid_search.best_params_.items()
    }
    print(f"\n  Best hyperparameters: {best_params}")
    print(f"  Best CV macro F1    : {grid_search.best_score_:.3f}")

    cv_df = pd.DataFrame(grid_search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    keep_cols = (
        [c for c in cv_df.columns if c.startswith("param_")]
        + ["mean_test_score", "std_test_score", "rank_test_score"]
    )
    cv_df = cv_df[keep_cols].rename(columns={
        "mean_test_score": "mean_cv_f1_macro",
        "std_test_score":  "std_cv_f1_macro",
    })
    cv_df.columns = [c.replace("param_classifier__", "") for c in cv_df.columns]
    cv_df = cv_df.round(4).reset_index(drop=True)

    return grid_search.best_estimator_, cv_df

def evaluate(
    pipeline: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Measure final performance on the test set.  This set was never used
    during training or hyperparameter tuning.

    PRIORITY METRICS FOR A FLOOD WARNING SYSTEM
    ─────────────────────────────────────────────
    1. High-class Recall      (most important)
       "Of all real High-risk events, what fraction did we catch?"
       Missing a High event = putting people in danger without warning.

    2. Moderate-class Recall
       "Of all real Moderate events, what fraction did we catch?"
       Missing Moderate events = people do not take precautions.

    3. Macro F1               (primary comparison metric)
       Balances precision and recall across all three classes equally.
       Use this to compare against the Logistic Regression baseline.

    We also report accuracy and the full confusion matrix, but these
    two class-level recall scores are what matter most for safety.
    """
    banner("STEP 6 – Evaluate on held-out test set")

    y_pred = pipeline.predict(X_test)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro",   labels=CLASS_ORDER, zero_division=0)
    macro_p  = precision_score(y_test, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
    macro_r  = recall_score(y_test, y_pred, average="macro",    labels=CLASS_ORDER, zero_division=0)

    print(f"\n  Accuracy        : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  Macro Precision : {macro_p:.3f}")
    print(f"  Macro Recall    : {macro_r:.3f}")
    print(f"  Macro F1        : {macro_f1:.3f}  ← PRIMARY METRIC")

    report_text = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    print("\n  Per-class report:")
    for line in report_text.splitlines():
        print(f"    {line}")

    report_dict = classification_report(
        y_test, y_pred, labels=CLASS_ORDER,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    high_recall = report_df.loc["High", "recall"]     if "High"     in report_df.index else 0.0
    mod_recall  = report_df.loc["Moderate", "recall"] if "Moderate" in report_df.index else 0.0
    print(f"\n  ── Safety-critical recall scores ──────────────────────────")
    print(f"  High-risk recall     : {high_recall:.3f}  (target ≥ 0.80)")
    print(f"  Moderate-risk recall : {mod_recall:.3f}  (target ≥ 0.70)")

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    print("\n  Confusion matrix (rows=true label, cols=predicted label):")
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_string())

    rf_step = pipeline.named_steps["classifier"]
    if hasattr(rf_step, "oob_score_"):
        print(f"\n  OOB accuracy (training data, internal estimate) : "
              f"{rf_step.oob_score_:.3f}")
        print("  NOTE: OOB is a rough internal check only.  The test-set score")
        print("  above is the official evaluation result.")

    return report_df, cm



def save_classification_report(report_df: pd.DataFrame, filepath: Path) -> None:
    """Save per-class precision / recall / F1 as a CSV."""
    report_df.to_csv(filepath)
    print(f"  Classification report → {filepath}")


def save_confusion_matrix_image(cm: np.ndarray, filepath: Path) -> None:
    """
    Save the confusion matrix as a labelled heatmap.

    HOW TO READ THE PLOT
    ────────────────────
    Each cell [row i, col j] = number of observations where
      true label = CLASS_ORDER[i]  AND  predicted label = CLASS_ORDER[j].
    The ideal model has large numbers on the main diagonal (top-left to
    bottom-right) and zeros everywhere else.

    For flood safety, pay special attention to the bottom row (High risk):
      • Bottom-right cell  (High → High) : correct High detections
      • Bottom-left/middle (High → Low/Moderate) : MISSED floods ← danger
    """

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "AquaAlert – Random Forest (Main Model)\nConfusion Matrix",
        fontsize=14, y=1.02,
    )

    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Raw counts"),
        (axes[1], cm_norm, ".2f",  "Row-normalised (recall per class)"),
    ]:
        sns.heatmap(
            data,
            annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            linewidths=0.5, ax=ax, vmin=0,
            vmax=(1 if fmt == ".2f" else None),
        )
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label",      fontsize=11)
        ax.set_title(title,              fontsize=11)

    fig.text(
        0.99, -0.02,
        "PROTOTYPE – NOT AN OFFICIAL FLOOD WARNING",
        ha="right", fontsize=7, color="grey",
    )
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix image → {filepath}")


def save_feature_importance(
    pipeline: ImbPipeline,
    feature_names: list[str],
    fi_png: Path,
    fi_csv: Path,
) -> pd.DataFrame:
    """
    Extract and plot the Random Forest feature importances.

    WHAT IS FEATURE IMPORTANCE?
    Each tree in the forest splits data at nodes to reduce class impurity
    (Gini impurity).  The "importance" of a feature is the total weighted
    reduction in impurity it caused, averaged across all trees.
    A higher importance = the model found that feature more useful for
    separating the three classes.

    LIMITATION (important for your report)
    Random Forest's built-in importance can overrate high-cardinality numeric
    features (many unique values) and underrate binary flags.
    For a more reliable estimate in future work, use permutation importance
    (sklearn.inspection.permutation_importance).
    PROTOTYPE ASSUMPTION.
    """
    rf         = pipeline.named_steps["classifier"]
    importances = rf.feature_importances_
    std         = np.std(
        [tree.feature_importances_ for tree in rf.estimators_], axis=0
    )

    fi_df = (
        pd.DataFrame({
            "feature":    feature_names,
            "importance": importances,
            "std":        std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["importance"] = fi_df["importance"].round(4)
    fi_df["std"]        = fi_df["std"].round(4)

    fi_df.to_csv(fi_csv, index=False)
    print(f"  Feature importances CSV → {fi_csv}")

    top = fi_df.head(TOP_N_FEATURES)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [
        "#1a6faf" if imp >= top["importance"].quantile(0.67) else
        "#5ba3d0" if imp >= top["importance"].quantile(0.33) else
        "#a8cce4"
        for imp in top["importance"]
    ]
    bars = ax.barh(
        top["feature"][::-1],      
        top["importance"][::-1],
        xerr=top["std"][::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "alpha": 0.6},
    )
    ax.set_xlabel("Mean decrease in Gini impurity", fontsize=11)
    ax.set_title(
        f"AquaAlert – Random Forest\nTop {TOP_N_FEATURES} Feature Importances",
        fontsize=13, pad=12,
    )
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.99, 0.01,
        "PROTOTYPE – NOT AN OFFICIAL FLOOD WARNING",
        ha="right", fontsize=7, color="grey",
    )
    fig.tight_layout()
    fig.savefig(fi_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance chart → {fi_png}")

    return fi_df


def save_cv_results(cv_df: pd.DataFrame, filepath: Path) -> None:
    """Save the GridSearchCV results table so the team can audit the tuning."""
    cv_df.to_csv(filepath, index=False)
    print(f"  CV results → {filepath}")


def save_model(pipeline: ImbPipeline, filepath: Path) -> None:
    """
    Persist the fitted pipeline (SMOTE + scaler + Random Forest) to disk.

    To use this model later on new observations:

        import joblib, pandas as pd
        pipeline = joblib.load("models/rf_pipeline_final.pkl")

        # new_data must be a DataFrame with the same feature columns
        # used during training (NOT the raw CSV columns).
        prediction = pipeline.predict(new_data)
        # returns e.g. ["Low", "Moderate", "High"]

        probabilities = pipeline.predict_proba(new_data)
        # returns a (n_rows × 3) array of class probabilities

    NOTE: joblib files are version-specific.  Always save requirements.txt.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"  Trained pipeline → {filepath}")


def compare_with_baseline(
    rf_report_df: pd.DataFrame,
    baseline_csv: Path,
    output_txt: Path,
) -> None:
    """
    Print and save a side-by-side comparison of Random Forest vs. the
    Logistic Regression baseline trained in train_baseline_lr.py.

    This comparison answers the key question for the capstone report:
    "Did the more complex model earn its complexity?"

    WHAT TO LOOK FOR
    ─────────────────────────────────────────────────────────────────────
    Macro F1       – did RF beat LR overall?
    High recall    – did RF miss fewer real flood events?
    High precision – did RF produce fewer false alarms?

    If RF improves all three, it clearly earns its place as the main model.
    If it only improves one, discuss the trade-off in your results section.
    """
    banner("STEP 8 – Compare with Logistic Regression baseline")

    if not baseline_csv.exists():
        print(f"  Baseline report not found at {baseline_csv}.")
        print("  Run train_baseline_lr.py first to generate it.")
        return

    lr_df = pd.read_csv(baseline_csv, index_col=0)

    metrics = ["precision", "recall", "f1-score"]
    rows_to_compare = CLASS_ORDER + ["macro avg"]

    lines = [
        "=" * 70,
        "  AquaAlert – Random Forest vs Logistic Regression Baseline",
        "  (Both models evaluated on the same time-based test set)",
        "=" * 70,
        "",
        f"  {'Metric':<35}  {'LR Baseline':>12}  {'Random Forest':>13}  {'Δ (RF−LR)':>10}",
        "  " + "─" * 66,
    ]

    all_deltas = []
    for row in rows_to_compare:
        for metric in metrics:
            label = f"{row} / {metric}"
            try:
                lr_val = float(lr_df.loc[row, metric])
                rf_val = float(rf_report_df.loc[row, metric])
                delta  = rf_val - lr_val
                all_deltas.append(delta)
                sign   = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "≈")
                lines.append(
                    f"  {label:<35}  {lr_val:>12.3f}  {rf_val:>13.3f}  "
                    f"{sign} {delta:+.3f}"
                )
            except KeyError:
                lines.append(f"  {label:<35}  {'N/A':>12}  {'N/A':>13}")
        lines.append("")

    mean_delta = np.mean(all_deltas) if all_deltas else 0
    verdict = (
        "RF outperforms the baseline overall (mean Δ > 0)."
        if mean_delta > 0 else
        "Baseline matches or outperforms RF overall.  Investigate before choosing RF."
    )
    lines += [
        "─" * 70,
        f"  Mean metric improvement (RF − LR) : {mean_delta:+.3f}",
        f"  Verdict : {verdict}",
        "",
        "  PROTOTYPE NOTICE",
        textwrap.fill(
            "Results are from real 2016–2020 daily NCR data with a 3-class "
            "label derived from rainfall and river-level thresholds.  "
            "They show how well each model recovers those thresholds, "
            "not validated flood-forecasting skill.",
            width=66, initial_indent="  ", subsequent_indent="  ",
        ),
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    print(report_text)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Comparison table saved → {output_txt}")


def interpret_results(
    report_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    cm: np.ndarray,
    y_test: pd.Series,
) -> None:
    """Print a simple explanation of the key results in plain language."""
    banner("STEP 9 – Plain-language interpretation")

    macro_f1    = float(report_df.loc["macro avg",  "f1-score"])
    high_recall = float(report_df.loc["High",       "recall"])   if "High"     in report_df.index else 0.0
    high_prec   = float(report_df.loc["High",       "precision"]) if "High"    in report_df.index else 0.0
    mod_recall  = float(report_df.loc["Moderate",   "recall"])   if "Moderate" in report_df.index else 0.0

    n_high_true    = int((y_test == "High").sum())
    n_high_correct = int(cm[CLASS_ORDER.index("High"), CLASS_ORDER.index("High")])

    top3_features = fi_df.head(3)["feature"].tolist()

    paragraphs = [
        f"How accurate is the model overall?  (Macro F1 = {macro_f1:.2f})",
        (
            f"The macro F1 of {macro_f1:.2f} means the model is broadly "
            f"{'good' if macro_f1 >= 0.80 else 'acceptable' if macro_f1 >= 0.70 else 'weak'} "
            f"at predicting all three flood-risk categories together.  Compare "
            f"this number against the Logistic Regression baseline: if it is "
            f"higher, the Random Forest earned its extra complexity."
        ),
        "",
        f"Does it catch real floods?  (High-risk recall = {high_recall:.2f})",
        (
            f"The model correctly identified {n_high_correct} out of {n_high_true} "
            f"truly High-risk observations in the test set "
            f"({high_recall * 100:.0f}% recall).  "
            f"{'This meets the ≥ 0.80 target.' if high_recall >= 0.80 else 'This is below the ≥ 0.80 target — investigate the missed events.'} "
            f"High-risk precision is {high_prec:.2f}, meaning roughly "
            f"{high_prec * 100:.0f}% of High alerts are correct "
            f"({'few false alarms' if high_prec >= 0.70 else 'a notable number of false alarms'})."
        ),
        "",
        f"What features drove the predictions most?",
        (
            f"The three most important features according to the Random Forest "
            f"are: {', '.join(top3_features)}.  "
            f"Higher importance means the model used that feature more often when "
            f"deciding which branch to follow in its decision trees.  "
            f"Check the feature importance chart and confirm that these features "
            f"make physical sense for NCR flood risk.  If a feature you expect to "
            f"be important (e.g. flood_hazard_level_enc) is ranked low, it may "
            f"be because the label is built from rainfall and river level only."
        ),
        "",
        "What are this model's limitations?",
        (
            "1. Threshold-derived label: the source flood label is exactly "
            "rainfall >= 30 mm, and Moderate is defined by rain/river thresholds, "
            "so near-perfect scores are expected and do not prove forecasting skill. "
            "2. No typhoon-scale events: the wettest day is ~53 mm, while major "
            "NCR floods (JICA 2014) had 149–291 mm/day. "
            "3. City-level, daily data: only four cities, one reading per day; "
            "no barangay detail or hourly bursts. "
            "4. Static hazard map: AEGIS points are assigned to cities by nearest "
            "city centre, not official boundaries. "
            "Document all four limitations in your capstone report."
        ),
        "",
        "PROTOTYPE NOTICE",
        (
            "This model is a student capstone prototype and produces "
            "DECISION-SUPPORT scores only.  It does NOT issue official flood "
            "warnings and does NOT replace PAGASA, LGU DRRMO, or NDRRMC advisories."
        ),
    ]

    print()
    for para in paragraphs:
        if para == "" or para.endswith("?") or para == "PROTOTYPE NOTICE":
            print(f"\n  {para}")
        else:
            print(textwrap.fill(para, width=63,
                                initial_indent="  ",
                                subsequent_indent="  "))

def main() -> None:
    print("\n" + "=" * 65)
    print("  AquaAlert – Random Forest (Main Model)")
    print("  Comparison target: Logistic Regression baseline")
    print("=" * 65)

    df = load_and_validate(DATA_FILE)

    X, y = build_feature_matrix(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = time_split(df, X, y)

    banner("STEP 4 – Build pipeline")
    pipeline = build_pipeline()
    print("  Pipeline steps:")
    for name, step in pipeline.steps:
        print(f"    [{name}]  {step.__class__.__name__}")

    best_pipeline, cv_df = tune_hyperparameters(pipeline, X_train, y_train)

    report_df, cm = evaluate(best_pipeline, X_test, y_test)

    banner("STEP 7 – Save reports and model")
    save_classification_report(report_df, CR_CSV)
    save_confusion_matrix_image(cm, CM_PNG)
    fi_df = save_feature_importance(best_pipeline, feature_names, FI_PNG, FI_CSV)
    save_cv_results(cv_df, CV_CSV)
    save_model(best_pipeline, MODEL_FILE)

    compare_with_baseline(report_df, LR_CR_CSV, CMP_TXT)

    interpret_results(report_df, fi_df, cm, y_test)

    banner("DONE – Output files")
    outputs = [MODEL_FILE, CR_CSV, CM_PNG, FI_PNG, FI_CSV, CV_CSV, CMP_TXT]
    for p in outputs:
        status = "✓" if p.exists() else "✗ NOT CREATED"
        print(f"  {status}  {p}")
    print()
    print("  PROTOTYPE NOTICE: outputs are for a student capstone project.")
    print("  They do NOT constitute official flood risk assessments.")
    print()
    print("  Next steps:")
    print("  1. Add independently recorded flood events (e.g. NDRRMC sitreps)")
    print("     so the label is not a rainfall threshold.")
    print("  2. Add typhoon-season data with > 100 mm/day rainfall.")
    print("  3. Document all prototype limitations in your capstone report.")


if __name__ == "__main__":
    main()
