"""

PROTOTYPE NOTICE
----------------
This is a student capstone prototype.  Its output does NOT
constitute an official flood warning and does NOT replace
advisories from PAGASA, LGU DRRMOs, or the NDRRMC.
All threshold and parameter choices are PROTOTYPE ASSUMPTIONS.

HOW TO RUN
----------
    python train_baseline_lr.py

INPUT  : data/processed/aquaalert_features.csv
OUTPUTS
    models/baseline_lr_pipeline.pkl       – saved trained pipeline
    reports/lr_classification_report.csv  – per-class precision/recall/F1
    reports/lr_confusion_matrix.png       – confusion matrix heatmap
    reports/lr_feature_weights.csv        – model coefficients per feature
"""

import warnings
import textwrap
from pathlib import Path

import joblib                          
import matplotlib
matplotlib.use("Agg")                 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")    

DATA_FILE    = Path("data/processed/aquaalert_features.csv")
MODEL_FILE   = Path("models/baseline_lr_pipeline.pkl")
REPORT_CSV   = Path("reports/lr_classification_report.csv")
CM_IMAGE     = Path("reports/lr_confusion_matrix.png")
WEIGHTS_CSV  = Path("reports/lr_feature_weights.csv")

CLASS_ORDER = ["Low", "Moderate", "High"]

TRAIN_RATIO = 0.70

LR_MAX_ITER = 1000

LR_SOLVER = "lbfgs"

LR_CLASS_WEIGHT = "balanced"

LR_C = 1.0

for folder in [MODEL_FILE.parent, REPORT_CSV.parent, CM_IMAGE.parent]:
    folder.mkdir(parents=True, exist_ok=True)


def banner(title: str) -> None:
    """Print a section divider to make console output easy to scan."""
    width = 65
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Read the preprocessed AquaAlert feature CSV.

    The timestamp column is parsed as a proper datetime so we can sort by it
    for the time-based train/test split in the next step.
    """
    banner("STEP 1 – Load dataset")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Cannot find {filepath}.\n"
            "Run preprocess.py first to generate the processed CSV."
        )

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    print(f"  Rows : {len(df):,}")
    print(f"  Cols : {len(df.columns)}")
    print(f"  Date range: {df['timestamp'].min().date()} → "
          f"{df['timestamp'].max().date()}")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into:
      X  – the input features (what the model learns from)
      y  – the target labels (what the model is trying to predict)

    TARGET LEAKAGE GUARD
    --------------------
    We explicitly list every column that must NOT be in X:
      • flood_risk      – the target label itself (text version)
      • flood_risk_enc  – numeric version of the target
      • observation_id  – just a row ID, carries no predictive meaning
      • timestamp       – raw datetime; we already extracted hour/month/dayofweek
      • barangay        – location text; needs special encoding (future work)
      • municipality    – same reason
      • flood_hazard_level / drainage_condition – text versions; the _enc
        columns already carry the same information as numbers

    Including any of these as model inputs would be target leakage or would
    cause errors.  If you are ever unsure whether a column is safe to include,
    ask: "Could I know this value BEFORE the flood event occurs?"
    """
    banner("STEP 2 – Separate features and target")

    EXCLUDE = {
        "flood_risk",         
        "flood_risk_enc",     
        "observation_id",
        "timestamp",
        "barangay",
        "municipality",
        "flood_hazard_level",  
        "drainage_condition",  
    }

    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[feature_cols].copy()
    y = df["flood_risk"].copy()

    print(f"  Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"    • {col}")
    print(f"\n  Target  : flood_risk")
    print(f"  Classes : {y.value_counts().to_dict()}")
    return X, y


def time_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sort all rows by timestamp and take the earliest TRAIN_RATIO fraction
    as the training set, the remainder as the test set.

    This mimics real-world model deployment:
      • Training set  = historical observations (the model learns from the past)
      • Test set      = more-recent observations (we measure how well it predicts
                        conditions it has never seen during training)

    A RANDOM split would be wrong here because it could mix future data into
    the training set and make accuracy look better than it really is.
    """
    banner("STEP 3 – Time-based train / test split")


    sorted_idx = df["timestamp"].argsort().values
    X_sorted   = X.iloc[sorted_idx].reset_index(drop=True)
    y_sorted   = y.iloc[sorted_idx].reset_index(drop=True)

    split_point = int(len(X_sorted) * train_ratio)

    X_train, X_test = X_sorted.iloc[:split_point], X_sorted.iloc[split_point:]
    y_train, y_test = y_sorted.iloc[:split_point], y_sorted.iloc[split_point:]

    print(f"  Train rows : {len(X_train):,}  ({100*len(X_train)/len(X_sorted):.0f}%)")
    print(f"  Test rows  : {len(X_test):,}  ({100*len(X_test)/len(X_sorted):.0f}%)")
    print()


    for name, y_part in [("Train", y_train), ("Test", y_test)]:
        counts = y_part.value_counts()
        dist   = ", ".join(f"{lbl}={counts.get(lbl,0)}" for lbl in CLASS_ORDER)
        print(f"  {name} class distribution: {dist}")

    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    """
    Assemble two steps into a single scikit-learn Pipeline object.

    STEP A  StandardScaler
    ──────────────────────
    Rescales every numeric feature to have mean = 0 and standard deviation = 1.
    Example:  rainfall_last_3h_mm might range 0–150 mm.
              elevation_m might range 3–50 m.
              Without scaling, the model treats large numbers as more important
              just because they are bigger — not because they are more informative.
    After scaling, every feature contributes on equal footing.

    WHY ONLY FIT ON TRAINING DATA?
    The scaler must learn mean/std from the TRAINING set only.  If we fitted it
    on the full dataset (including the test set), the scaler would "peek" at
    future data — another form of data leakage.  The Pipeline object handles
    this automatically: when you call pipeline.fit(X_train, y_train), the
    scaler learns from X_train; when you call pipeline.predict(X_test), it
    transforms X_test using the training mean/std without refitting.

    STEP B  LogisticRegression
    ──────────────────────────
    See the configuration section at the top of this file for hyperparameter
    explanations (max_iter, solver, multi_class, class_weight, C).
    """
    scaler = StandardScaler()

    classifier = LogisticRegression(

        solver=LR_SOLVER,           
        max_iter=LR_MAX_ITER,         
        class_weight=LR_CLASS_WEIGHT,  
        C=LR_C,                       
        random_state=42,               
    )

    pipeline = Pipeline(steps=[
        ("scaler",     scaler),
        ("classifier", classifier),
    ])
    return pipeline


def train(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Fit (train) the pipeline on the training data.

    pipeline.fit() does two things in order:
      1. Fits the scaler on X_train (learns mean and std for each feature).
      2. Transforms X_train using those statistics.
      3. Trains the Logistic Regression on the scaled training data.
    """
    banner("STEP 5 – Train Logistic Regression")
    print("  Training … ", end="", flush=True)
    pipeline.fit(X_train, y_train)
    print("done.")
    return pipeline


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Measure how well the trained model predicts the test set.

    METRICS EXPLAINED (for college students)
    -----------------------------------------
    Accuracy
        Of all predictions, what fraction were correct?
        Problem: with imbalanced classes, a model that always predicts
        "Moderate" gets 51 % accuracy without ever catching a flood.

    Precision (per class)
        Of all rows the model labelled "High", what fraction actually were High?
        High precision = few false alarms.

    Recall (per class)
        Of all rows that truly were "High", what fraction did the model catch?
        High recall = few missed floods.
        For a flood warning system, HIGH RECALL ON THE HIGH CLASS IS THE MOST
        IMPORTANT METRIC.  Missing a real High-risk event is more dangerous
        than a false alarm.

    F1-score (per class)
        The harmonic mean of precision and recall.
        F1 = 2 × (precision × recall) / (precision + recall)
        It balances both concerns into a single number.

    Macro F1
        Average F1 across all three classes (Low, Moderate, High), treating
        each class equally regardless of how many rows it has.
        This is our PRIMARY EVALUATION METRIC because it penalises the model
        for doing badly on the rare High class.

    Confusion Matrix
        A 3×3 grid.  Row = true label.  Column = predicted label.
        The diagonal cells = correct predictions.
        Off-diagonal cells = errors (what the model confused with what).
    """
    banner("STEP 6 – Evaluate on test set")

    y_pred = pipeline.predict(X_test)

    accuracy   = accuracy_score(y_test, y_pred)
    macro_f1   = f1_score(y_test, y_pred, average="macro",   labels=CLASS_ORDER, zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
    macro_rec  = recall_score(y_test, y_pred, average="macro",    labels=CLASS_ORDER, zero_division=0)

    print(f"\n  Accuracy          : {accuracy:.3f}  ({accuracy*100:.1f}%)")
    print(f"  Macro Precision   : {macro_prec:.3f}")
    print(f"  Macro Recall      : {macro_rec:.3f}")
    print(f"  Macro F1-score    : {macro_f1:.3f}  ← PRIMARY METRIC")
    print()

    report_text = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    print("  Per-class report:")
    for line in report_text.splitlines():
        print(f"    {line}")

    report_dict = classification_report(
        y_test, y_pred, labels=CLASS_ORDER,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    print("\n  Confusion matrix (rows = true, cols = predicted):")
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
    print(cm_df.to_string())

    return report_df, cm

def save_classification_report(report_df: pd.DataFrame, filepath: Path) -> None:
    """Save the per-class precision/recall/F1 table to a CSV file."""
    report_df.to_csv(filepath)
    print(f"\n  Classification report saved → {filepath}")


def save_confusion_matrix_image(cm: np.ndarray, filepath: Path) -> None:
    """
    Save the confusion matrix as a colour-coded heatmap PNG.

    The colour scale runs white (0 predictions) → dark blue (many predictions).
    Ideal model: dark diagonal, white everywhere else.
    For a flood system, pay attention to the bottom row (true High):
      if the rightmost cell (predicted High) is dark, the model is catching floods.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,           
        fmt="d",             
        cmap="Blues",
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(
        "AquaAlert – Logistic Regression (Baseline)\nConfusion Matrix",
        fontsize=13, pad=14,
    )

    fig.text(
        0.99, 0.01,
        "PROTOTYPE – NOT AN OFFICIAL FLOOD WARNING",
        ha="right", va="bottom", fontsize=7, color="grey",
    )

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix image saved → {filepath}")


def save_feature_weights(
    pipeline: Pipeline,
    feature_names: list[str],
    filepath: Path,
) -> None:
    """
    Extract and save the Logistic Regression coefficients (weights).

    WHY ARE COEFFICIENTS USEFUL?
    Each coefficient tells us: "When this feature goes up by 1 standard
    deviation, how much does the log-probability of a given class go up or down?"
    A large positive coefficient for 'rainfall_last_3h_mm → High' means the
    model learned that heavy short-burst rainfall strongly predicts High risk.
    Negative coefficients push the model away from a class.

    This table is useful for:
    - Verifying that the model learned sensible patterns (a sanity check).
    - Explaining the model's reasoning to stakeholders.
    - Comparing with the Random Forest's feature importances later.
    """
    clf    = pipeline.named_steps["classifier"]
    coefs  = clf.coef_         
    labels = clf.classes_      

    rows = []
    for i, cls in enumerate(labels):
        for j, feat in enumerate(feature_names):
            rows.append({
                "class":       cls,
                "feature":     feat,
                "coefficient": round(float(coefs[i, j]), 4),
            })

    weights_df = (
        pd.DataFrame(rows)
        .sort_values(["class", "coefficient"], ascending=[True, False])
    )
    weights_df.to_csv(filepath, index=False)
    print(f"  Feature weights saved → {filepath}")



def save_pipeline(pipeline: Pipeline, filepath: Path) -> None:
    """
    Persist the fitted pipeline to disk using joblib.

    The saved .pkl file contains BOTH the fitted scaler AND the trained
    classifier in one object.  To make a prediction on new data later:

        import joblib
        pipeline = joblib.load("models/baseline_lr_pipeline.pkl")
        # new_row is a DataFrame with the same feature columns
        prediction = pipeline.predict(new_row)

    IMPORTANT: the .pkl file is specific to this Python and scikit-learn
    version.  Always save requirements.txt alongside it.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"\n  Trained pipeline saved → {filepath}")


def interpret_results(
    report_df: pd.DataFrame,
    cm: np.ndarray,
    y_test: pd.Series,
) -> None:
    """
    Print a simple explanation of what the numbers mean so team members who
    are not data scientists can understand the model's performance.
    """
    banner("STEP 9 – Plain-language interpretation")

    macro_f1 = float(report_df.loc["macro avg", "f1-score"])


    high_recall   = float(report_df.loc["High", "recall"])   if "High"     in report_df.index else 0.0
    high_prec     = float(report_df.loc["High", "precision"]) if "High"    in report_df.index else 0.0
    low_recall    = float(report_df.loc["Low",  "recall"])    if "Low"     in report_df.index else 0.0
    mod_recall    = float(report_df.loc["Moderate","recall"]) if "Moderate" in report_df.index else 0.0


    n_high_true = int((y_test == "High").sum())

    n_high_correct = int(cm[CLASS_ORDER.index("High"), CLASS_ORDER.index("High")])

    paragraphs = [
        f"Overall quality (Macro F1 = {macro_f1:.2f})",
        (
            f"The macro F1-score is {macro_f1:.2f} out of 1.00.  This means the "
            f"model does a roughly {macro_f1*100:.0f}% good job balancing its "
            f"performance across all three flood-risk classes.  Scores above 0.70 "
            f"are considered acceptable for a baseline; above 0.80 is good.  "
            f"This score was measured on the test set (rows the model never saw "
            f"during training), so it reflects real-world generalisation, not "
            f"memorisation."
        ),
        "",
        f"High-risk detection (Recall = {high_recall:.2f})",
        (
            f"Out of {n_high_true} truly High-risk observations in the test set, "
            f"the model correctly identified {n_high_correct} "
            f"({high_recall*100:.0f}%).  This is the most safety-critical number "
            f"for a flood warning system.  A missed High-risk event is more "
            f"dangerous than a false alarm, so we want this number to be as high "
            f"as possible, ideally above 0.80."
        ),
        (
            f"The precision for High-risk is {high_prec:.2f}, meaning that "
            f"{high_prec*100:.0f}% of the time the model flags High risk, it is "
            f"correct.  Lower precision = more false alarms."
        ),
        "",
        "Why does Logistic Regression sometimes struggle with rare classes?",
        (
            "The High-risk class makes up only about 6% of the dataset.  Even "
            "with class_weight='balanced', a linear model like Logistic Regression "
            "can find it hard to capture the non-linear combinations of signals "
            "(e.g. high rainfall AND blocked drain AND rising river) that together "
            "indicate High risk.  This is one reason we will train a Random Forest "
            "next: it can learn those interactions naturally."
        ),
        "",
        "What to do with this baseline",
        (
            "Record the macro F1, High-recall, and High-precision numbers in your "
            "capstone report.  When you train the Random Forest, compare those same "
            "three numbers.  If the Random Forest improves on all three, that "
            "justifies using the more complex model.  If it only improves on one, "
            "discuss the trade-off in your results section."
        ),
        "",
        "PROTOTYPE NOTICE",
        (
            "All results above come from a synthetic dataset generated for "
            "prototype testing.  They do NOT reflect real flood patterns in NCR.  "
            "Collect real labelled observations before drawing conclusions about "
            "the AquaAlert system's actual performance."
        ),
    ]

    print()
    for para in paragraphs:
        if para == "" or para.endswith(")"):
            print(f"\n  {para}")
        else:

            wrapped = textwrap.fill(para, width=63,
                                    initial_indent="  ",
                                    subsequent_indent="  ")
            print(wrapped)


def main() -> None:
    print("\n" + "=" * 65)
    print("  AquaAlert – Logistic Regression Baseline")
    print("  (Comparison baseline; not a filtering model)")
    print("=" * 65)
    print(f"\n  Input  : {DATA_FILE}")
    print(f"  Model  : {MODEL_FILE}")
    print(f"  Report : {REPORT_CSV}")
    print(f"  Plot   : {CM_IMAGE}")

    df = load_data(DATA_FILE)

    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = time_split(df, X, y, TRAIN_RATIO)


    banner("STEP 4 – Build pipeline")
    pipeline = build_pipeline()
    print("  Pipeline steps:")
    for name, step in pipeline.steps:
        print(f"    [{name}]  {step}")

    train(pipeline, X_train, y_train)

    report_df, cm = evaluate(pipeline, X_test, y_test)

    banner("STEP 7 – Save reports and model")
    save_classification_report(report_df, REPORT_CSV)
    save_confusion_matrix_image(cm, CM_IMAGE)
    save_feature_weights(pipeline, list(X.columns), WEIGHTS_CSV)
    save_pipeline(pipeline, MODEL_FILE)

    interpret_results(report_df, cm, y_test)

    banner("DONE – Output files")
    outputs = [MODEL_FILE, REPORT_CSV, CM_IMAGE, WEIGHTS_CSV]
    for p in outputs:
        status = "✓" if p.exists() else "✗ NOT CREATED"
        print(f"  {status}  {p}")
    print()
    print("  Next step: run train_random_forest.py and compare its")
    print("  macro F1 and High-class recall against the numbers above.")
    print()
    print("  PROTOTYPE NOTICE: outputs are for a student capstone project.")
    print("  They do NOT constitute official flood risk assessments.")


if __name__ == "__main__":
    main()
