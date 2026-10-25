"""
preprocess.py
=============
AquaAlert – Flood-Risk Dataset Preprocessing Pipeline
------------------------------------------------------
WHAT THIS SCRIPT DOES
---------------------
Takes the raw AquaAlert CSV, cleans it up, builds new features that help
the machine-learning model learn better, and saves the result as a clean
CSV ready for training.

HOW TO RUN
----------
    python preprocess.py

INPUT  : data/raw/aquaalert_raw.csv
OUTPUT : data/processed/aquaalert_features.csv
REPORT : reports/data_quality_report.txt

EXPECTED COLUMNS IN THE RAW FILE
---------------------------------
  observation_id              – unique row ID (e.g. "OBS-0001")
  timestamp                   – date/time of observation (ISO 8601 preferred)
  barangay                    – name of the barangay
  municipality                – city/municipality name
  rainfall_last_3h_mm         – cumulative rainfall in the last 3 hours (mm)
  rainfall_last_24h_mm        – cumulative rainfall in the last 24 hours (mm)
  forecast_rainfall_next_3h_mm– weather-forecast rainfall for the next 3 h (mm)
  river_water_level_m         – river level in metres (can be blank/NaN)
  water_level_change_1h_m     – rise/fall in river level over 1 h (can be blank)
  elevation_m                 – barangay elevation in metres
  flood_hazard_level          – government-assessed zone: Low / Medium / High
  drainage_condition          – observed drain state: Clean / Partly Blocked / Blocked
  citizen_report_count        – number of flood reports in the last 6 h
  average_reported_flood_depth_cm – avg depth from citizen reports (cm)
  flood_risk                  – TARGET LABEL: Low / Moderate / High

PROTOTYPE NOTICE
----------------
This script is part of a student capstone prototype. It does not produce
official flood warnings and does not replace PAGASA, LGU, or NDRRMC advisories.
All threshold values used here are prototype assumptions that need local validation.

Authors : AquaAlert Capstone Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# =============================================================================
# CONFIGURATION – change paths here if your folder structure is different
# =============================================================================

RAW_FILE    = Path("data/raw/aquaalert_raw.csv")
OUTPUT_FILE = Path("data/processed/aquaalert_features.csv")
REPORT_FILE = Path("reports/data_quality_report.txt")

# These are the only numeric input columns we will keep after preprocessing.
# "flood_risk" (the target) is handled separately so it is never accidentally
# used as a feature – that would be target leakage.
NUMERIC_INPUT_COLS = [
    "rainfall_last_3h_mm",
    "rainfall_last_24h_mm",
    "forecast_rainfall_next_3h_mm",
    "river_water_level_m",          # allowed to be missing (NaN)
    "water_level_change_1h_m",      # allowed to be missing (NaN)
    "elevation_m",
    "citizen_report_count",
    "average_reported_flood_depth_cm",
]

# Numeric columns that MUST have a value (no NaN allowed after imputation).
REQUIRED_NUMERIC_COLS = [
    "rainfall_last_3h_mm",
    "rainfall_last_24h_mm",
    "forecast_rainfall_next_3h_mm",
    "elevation_m",
    "citizen_report_count",
    "average_reported_flood_depth_cm",
]

# These two are allowed to be missing because not every barangay has a river
# sensor nearby.  We fill them with a special sentinel value (−999) and add a
# binary "was_missing" flag so the model can learn from the absence of data.
SENSOR_COLS_NULLABLE = ["river_water_level_m", "water_level_change_1h_m"]

# Valid category sets – used for validation and encoding
VALID_HAZARD_LEVELS    = ["Low", "Medium", "High"]     # ordered Low → High
VALID_DRAINAGE_CONDITIONS = ["Clean", "Partly Blocked", "Blocked"]  # ordered
VALID_RISK_LABELS      = ["Low", "Moderate", "High"]   # ordered Low → High

# The sentinel value written when a sensor reading is absent.
# Using −999 (not 0 or NaN) so the model can distinguish "no sensor" from
# "sensor reads zero".  This is a PROTOTYPE ASSUMPTION; consider a dedicated
# indicator column instead if your model supports it.
MISSING_SENSOR_SENTINEL = -999.0

# Thresholds for the engineered "water_level_rise_category" feature.
# These match the scorer's PROTOTYPE ASSUMPTIONS – validate against EFCOS data.
RISE_THRESHOLD_RAPID_M_PER_H      = 1.5
RISE_THRESHOLD_VERY_RAPID_M_PER_H = 2.5

# Rainfall intensity thresholds (PAGASA-aligned, PROTOTYPE ASSUMPTION).
RAIN_3H_MODERATE_MM = 50.0
RAIN_3H_HIGH_MM     = 100.0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def banner(title: str) -> None:
    """Print a section header to the console so progress is easy to read."""
    width = 65
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Read the raw CSV into a pandas DataFrame.

    pandas will automatically detect column types, but we tell it to treat
    'observation_id' as a plain string (not a number) so IDs like 'OBS-001'
    are preserved exactly.
    """
    banner("STEP 1 – Load raw data")

    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        print("  Make sure aquaalert_raw.csv is in the data/raw/ folder.")
        sys.exit(1)

    df = pd.read_csv(
        filepath,
        dtype={"observation_id": str},   # keep IDs as strings
        low_memory=False,                 # avoids mixed-type warnings on large files
    )

    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'timestamp' column to a proper datetime type and extract
    useful time-based features.

    Why do this?
    - pandas stores dates as plain text by default; converting them lets us
      do time-based math (e.g., how many hours apart are two rows?).
    - Extracting hour and month helps the model learn daily and seasonal
      rainfall patterns (typhoon season in NCR is roughly June–November).
    """
    banner("STEP 2 – Parse timestamps")

    if "timestamp" not in df.columns:
        print("  [WARNING] No 'timestamp' column found – skipping this step.")
        return df

    # Convert text to datetime.  'errors=coerce' turns unparseable values to NaT
    # (Not a Time) instead of crashing.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    n_bad = df["timestamp"].isna().sum()
    if n_bad > 0:
        print(f"  [WARNING] {n_bad} rows had unparseable timestamps → set to NaT")

    # Extract useful sub-fields.
    # These give the model access to seasonal and time-of-day signals without
    # passing the raw timestamp (which would be a different value for every row
    # and would confuse the model).
    df["obs_hour"]       = df["timestamp"].dt.hour          # 0–23
    df["obs_month"]      = df["timestamp"].dt.month         # 1–12
    df["obs_dayofweek"]  = df["timestamp"].dt.dayofweek     # 0=Monday, 6=Sunday
    df["is_typhoon_season"] = df["obs_month"].isin(
        [6, 7, 8, 9, 10, 11]
    ).astype(int)  # 1 during Jun–Nov typhoon/monsoon season in the Philippines

    print(f"  Timestamp range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print("  Added columns: obs_hour, obs_month, obs_dayofweek, is_typhoon_season")
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows that are exact duplicates.

    Why?
    - Duplicate rows can happen if data from multiple sources is merged
      carelessly.  They inflate the training set without adding new information,
      and they can make the model seem more accurate than it really is because
      the same row might appear in both training and test sets.

    We keep the FIRST occurrence and drop the rest.
    """
    banner("STEP 3 – Remove duplicates")

    rows_before = len(df)

    # Check for fully identical rows first
    n_full_dupes = df.duplicated().sum()

    # Also check for rows with the same observation_id (different content could
    # mean a data-entry error; we still drop the later one and warn the user)
    n_id_dupes = 0
    if "observation_id" in df.columns:
        n_id_dupes = df.duplicated(subset=["observation_id"]).sum()

    df = df.drop_duplicates().reset_index(drop=True)

    rows_removed = rows_before - len(df)
    print(f"  Fully identical duplicate rows removed : {n_full_dupes}")
    print(f"  Duplicate observation_id rows found    : {n_id_dupes}")
    print(f"  Rows remaining after deduplication     : {len(df):,}")
    return df, rows_removed


def validate_and_fix_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Check that categorical columns contain only expected values.
    Rows with invalid categories are flagged so we can decide what to do with
    them (here we drop them and report how many were removed).

    Why validate categories?
    - If someone accidentally typed "high" instead of "High" or "Partly blocked"
      instead of "Partly Blocked", the encoder will treat it as an unknown
      category and either crash or silently produce wrong numbers.
    """
    banner("STEP 4a – Validate categorical columns")

    validation_log = {}
    rows_to_drop = pd.Series(False, index=df.index)  # start: no rows flagged

    checks = {
        "flood_hazard_level":   VALID_HAZARD_LEVELS,
        "drainage_condition":   VALID_DRAINAGE_CONDITIONS,
        "flood_risk":           VALID_RISK_LABELS,  # the target label
    }

    for col, valid_values in checks.items():
        if col not in df.columns:
            print(f"  [WARNING] Column '{col}' not found – skipping validation.")
            validation_log[col] = {"missing_col": True}
            continue

        # Count invalid values (not in the expected set AND not NaN)
        is_invalid = ~df[col].isin(valid_values) & df[col].notna()
        n_invalid  = is_invalid.sum()
        validation_log[col] = {
            "valid_values": valid_values,
            "n_invalid_rows": int(n_invalid),
        }
        if n_invalid > 0:
            bad_vals = df.loc[is_invalid, col].unique().tolist()
            print(f"  [WARNING] '{col}': {n_invalid} rows with invalid values "
                  f"{bad_vals} → will be dropped")
            rows_to_drop |= is_invalid
        else:
            print(f"  '{col}': all values valid ✓")

    n_dropped = rows_to_drop.sum()
    if n_dropped > 0:
        df = df[~rows_to_drop].reset_index(drop=True)
        print(f"  Rows dropped due to invalid categories: {n_dropped}")
    return df, validation_log


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fill in missing values so the model doesn't crash on NaN inputs.

    Strategy by column type:
    ─────────────────────────────────────────────────────────────────────────
    Required numeric columns  → fill with the column MEDIAN (robust to outliers)
    River sensor columns      → fill with −999 sentinel + add a binary flag column
    Categorical columns       → fill with the column MODE (most common value)
    Target label (flood_risk) → rows with missing target are DROPPED (no choice;
                                 we cannot train on rows without a known answer)
    ─────────────────────────────────────────────────────────────────────────
    WHY MEDIAN, NOT MEAN?
    Flood data often contains extreme outliers (e.g., typhoon events with very
    high rainfall).  The median is not pulled toward those extremes the way the
    mean is, so it is a safer fill value for normal-day rows with missing data.
    """
    banner("STEP 4b – Handle missing values")

    imputation_log = {}

    # --- Drop rows where the target label is missing ---
    if "flood_risk" in df.columns:
        n_missing_target = df["flood_risk"].isna().sum()
        if n_missing_target > 0:
            print(f"  Dropping {n_missing_target} rows with missing flood_risk "
                  f"(cannot train without a label)")
            df = df[df["flood_risk"].notna()].reset_index(drop=True)
        imputation_log["flood_risk"] = {"rows_dropped": int(n_missing_target)}

    # --- River sensor columns: sentinel fill + missing indicator ---
    for col in SENSOR_COLS_NULLABLE:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        flag_col  = f"{col}_missing"                    # new binary column

        df[flag_col] = df[col].isna().astype(int)       # 1 = was NaN, 0 = had data
        df[col]      = df[col].fillna(MISSING_SENSOR_SENTINEL)

        imputation_log[col] = {
            "n_missing": int(n_missing),
            "strategy": f"sentinel={MISSING_SENSOR_SENTINEL}, flag_col={flag_col}",
        }
        print(f"  '{col}': {n_missing} missing → filled with {MISSING_SENSOR_SENTINEL},"
              f" added '{flag_col}' flag column")

    # --- Required numeric columns: median fill ---
    for col in REQUIRED_NUMERIC_COLS:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)
            imputation_log[col] = {
                "n_missing": int(n_missing),
                "strategy": "median",
                "fill_value": float(median_val),
            }
            print(f"  '{col}': {n_missing} missing → filled with median "
                  f"({median_val:.3f})")
        else:
            imputation_log[col] = {"n_missing": 0}

    # --- Categorical columns: mode fill ---
    cat_cols = ["flood_hazard_level", "drainage_condition", "barangay", "municipality"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]               # mode() returns a Series
            df[col]  = df[col].fillna(mode_val)
            imputation_log[col] = {
                "n_missing": int(n_missing),
                "strategy": "mode",
                "fill_value": mode_val,
            }
            print(f"  '{col}': {n_missing} missing → filled with mode ('{mode_val}')")
        else:
            imputation_log[col] = {"n_missing": 0}

    return df, imputation_log


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical text columns into numbers that the model can use.

    We use ORDINAL encoding (not one-hot encoding) because both columns have
    a natural order that matters:
      flood_hazard_level : Low < Medium < High       → encoded as 0, 1, 2
      drainage_condition : Clean < Partly Blocked < Blocked → encoded as 0, 1, 2

    WHY NOT ONE-HOT ENCODING?
    One-hot would create three new columns per variable and treat the categories
    as having no order between them.  But "Blocked" drainage is strictly worse
    than "Partly Blocked", so encoding that order is more informative for a
    model like Logistic Regression or Random Forest.

    The original text columns are kept alongside the encoded ones so you can
    still read the data easily.
    """
    banner("STEP 5 – Encode categorical features")

    # OrdinalEncoder needs a list-of-lists: one inner list per column,
    # giving the categories IN ORDER from lowest to highest.
    encoder = OrdinalEncoder(
        categories=[
            VALID_HAZARD_LEVELS,         # for flood_hazard_level
            VALID_DRAINAGE_CONDITIONS,   # for drainage_condition
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,                # unknown category → −1 (safe fallback)
    )

    cols_to_encode = ["flood_hazard_level", "drainage_condition"]
    # Only encode columns that actually exist in the dataframe
    cols_present = [c for c in cols_to_encode if c in df.columns]

    if cols_present:
        encoded_array = encoder.fit_transform(df[cols_present])
        for i, col in enumerate(cols_present):
            new_col_name = f"{col}_enc"
            df[new_col_name] = encoded_array[:, i].astype(int)
            print(f"  '{col}' → '{new_col_name}': "
                  f"{dict(zip(encoder.categories_[i], range(len(encoder.categories_[i]))))}")

    # Encode the TARGET label separately using a simple map so we keep control.
    # We keep the original text label too – the numeric version is for convenience.
    if "flood_risk" in df.columns:
        risk_map = {"Low": 0, "Moderate": 1, "High": 2}
        df["flood_risk_enc"] = df["flood_risk"].map(risk_map)
        print(f"  'flood_risk' → 'flood_risk_enc': {risk_map}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns (features) that combine or transform existing ones.
    Good features help the model find patterns that raw numbers alone might miss.

    Features created here:
    ──────────────────────────────────────────────────────────────────────────
    1. rainfall_intensity_3h_category
       Classifies 3-hour rainfall into Low / Moderate / High intensity buckets
       using PAGASA-aligned thresholds (PROTOTYPE ASSUMPTION).
       Why: the model may find it easier to learn from a 3-level category
       than to draw its own boundaries on a continuous number.

    2. rainfall_ratio_3h_to_24h
       Ratio of 3-hour to 24-hour rainfall (clipped to [0, 1]).
       Why: a high ratio means most of the day's rainfall fell in just 3 hours
       – a sudden intense burst that overwhelms drainage faster than a steady
       rain of the same total.
       Example: 3h=80 mm, 24h=100 mm → ratio=0.80 (burst event, high risk)
                3h=10 mm, 24h=100 mm → ratio=0.10 (steady rain, lower burst risk)

    3. water_level_rise_category
       Classifies the rate of river rise into:
         0 = No sensor / falling / stable
         1 = Rising (slow)
         2 = Rising rapidly   (≥ 1.5 m/hr – PROTOTYPE ASSUMPTION)
         3 = Rising very rapidly (≥ 2.5 m/hr – PROTOTYPE ASSUMPTION)
       Why: a river at 18 m rising at 2 m/hr is more dangerous than one at
       20 m rising at 0.1 m/hr because the first is accelerating toward danger.

    4. report_density_per_km2
       Citizen report count divided by barangay area in km² (if 'area_km2' exists)
       or just the raw count (if area is not available).
       Why: 10 reports in a 0.5 km² barangay is much more alarming than 10
       reports spread over 10 km².  Normalising by area removes the size effect.

    5. combined_rain_drainage_risk
       Multiplies the rainfall component score (0–1) by the drainage obstruction
       level (0–1) to get a single "blocked-drain-amplified-rainfall" number.
       Why: heavy rain on clean drains is less dangerous than moderate rain on
       fully blocked drains.  This feature encodes that interaction directly so
       the model does not have to discover it on its own.

    TARGET LEAKAGE CHECK
    ────────────────────
    None of these features use the 'flood_risk' label column.
    They are all derived from input observations only.
    Using the target to create a feature is called "target leakage" and causes
    a model to look artificially accurate during training but fail in the real world.
    """
    banner("STEP 6 – Engineer new features")

    # ------------------------------------------------------------------
    # Feature 1: Rainfall intensity category (3-hour window)
    # ------------------------------------------------------------------
    if "rainfall_last_3h_mm" in df.columns:
        def classify_rain_intensity(mm: float) -> int:
            """
            Map a 3-hour rainfall total to a category number.
            0 = Low   (< 50 mm / 3 h)
            1 = Moderate (50–100 mm / 3 h)
            2 = High  (≥ 100 mm / 3 h)
            Thresholds are PROTOTYPE ASSUMPTIONS aligned to PAGASA rainfall advisories.
            """
            if mm >= RAIN_3H_HIGH_MM:
                return 2    # High intensity
            elif mm >= RAIN_3H_MODERATE_MM:
                return 1    # Moderate intensity
            else:
                return 0    # Low intensity

        df["rainfall_intensity_3h_cat"] = df["rainfall_last_3h_mm"].apply(
            classify_rain_intensity
        )
        print("  + rainfall_intensity_3h_cat  (0=Low, 1=Moderate, 2=High)")

    # ------------------------------------------------------------------
    # Feature 2: Ratio of 3-hour to 24-hour rainfall (burst indicator)
    # ------------------------------------------------------------------
    if "rainfall_last_3h_mm" in df.columns and "rainfall_last_24h_mm" in df.columns:
        # Avoid division by zero: if 24h total is 0, ratio is 0
        # clip(0, 1) keeps the value between 0 and 1 (ratio cannot exceed 1
        # because 3h rainfall cannot exceed 24h total in a consistent dataset)
        df["rainfall_ratio_3h_24h"] = (
            df["rainfall_last_3h_mm"]
            / df["rainfall_last_24h_mm"].replace(0, np.nan)  # 0 → NaN to avoid /0
        ).fillna(0).clip(0, 1)

        print("  + rainfall_ratio_3h_24h      (0=steady rain, 1=all in 3h burst)")

    # ------------------------------------------------------------------
    # Feature 3: Water-level rise category
    # ------------------------------------------------------------------
    if "water_level_change_1h_m" in df.columns:
        def classify_rise(change: float) -> int:
            """
            Classify rate of water level change into a 4-level category.
            Sentinel value −999 means no sensor; those rows get category 0.
            PROTOTYPE ASSUMPTION: thresholds should be validated against EFCOS data.
            """
            if change == MISSING_SENSOR_SENTINEL or change <= 0:
                return 0    # No sensor data, falling, or stable
            elif change >= RISE_THRESHOLD_VERY_RAPID_M_PER_H:
                return 3    # Very rapid rise ≥ 2.5 m/hr
            elif change >= RISE_THRESHOLD_RAPID_M_PER_H:
                return 2    # Rapid rise ≥ 1.5 m/hr
            else:
                return 1    # Slow rise

        df["water_level_rise_cat"] = df["water_level_change_1h_m"].apply(classify_rise)
        print("  + water_level_rise_cat       (0=stable/no data, 1=slow, "
              "2=rapid, 3=very rapid)")

    # ------------------------------------------------------------------
    # Feature 4: Nearby-report density
    # ------------------------------------------------------------------
    if "citizen_report_count" in df.columns:
        if "area_km2" in df.columns:
            # Normalise by barangay area if available
            df["report_density_per_km2"] = (
                df["citizen_report_count"]
                / df["area_km2"].replace(0, np.nan)
            ).fillna(0)
            print("  + report_density_per_km2     (reports normalised by barangay area)")
        else:
            # Fall back to raw count; rename for consistency so downstream code
            # always finds the same column name
            df["report_density_per_km2"] = df["citizen_report_count"].astype(float)
            print("  + report_density_per_km2     (area_km2 not found – using raw count)")

    # ------------------------------------------------------------------
    # Feature 5: Combined rainfall × drainage risk
    # ------------------------------------------------------------------
    rain_col    = "rainfall_last_3h_mm"
    drain_col   = "drainage_condition_enc"  # created in encode_categoricals()

    if rain_col in df.columns and drain_col in df.columns:
        # Scale rainfall to [0, 1] using the HIGH threshold as the ceiling.
        # Clip at 1 so extreme rainfall doesn't push the product above 1.
        rain_scaled  = (df[rain_col] / RAIN_3H_HIGH_MM).clip(0, 1)

        # drainage_condition_enc is 0 (Clean) / 1 (Partly Blocked) / 2 (Blocked).
        # Divide by 2 to scale to [0, 1].
        drain_scaled = df[drain_col] / 2.0

        df["combined_rain_drainage_risk"] = rain_scaled * drain_scaled
        print("  + combined_rain_drainage_risk (0=no risk, 1=extreme rain + blocked drain)")

    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that must NOT be used as model input features.

    TARGET LEAKAGE means accidentally including information that directly
    reveals the answer (flood_risk) during training.  Examples of leaky columns:
    - flood_risk itself (the label – this is what we're trying to predict!)
    - flood_risk_enc (the numeric version of the label)
    - Any column derived from the label

    We keep 'flood_risk' in the output CSV but clearly label it as the target.
    The model-training script should load it separately as y, not as part of X.

    We also drop columns that carry no predictive signal for a model:
    - observation_id : just a row ID, not a real measurement
    - timestamp      : raw timestamp is too granular; we already extracted
                       hour/month/dayofweek features from it
    - barangay, municipality : free-text location names; these need special
      treatment (e.g. target encoding) before using in a model – for now we drop
      them from the feature matrix but keep them in the CSV for reference
    """
    banner("STEP 7 – Drop non-feature / leakage columns")

    # Columns to exclude from the feature matrix (kept in CSV as metadata/target)
    non_feature_cols = [
        "observation_id",     # row ID
        "timestamp",          # replaced by obs_hour / obs_month / obs_dayofweek
        "barangay",           # location text – needs further encoding
        "municipality",       # location text – needs further encoding
        "flood_risk",         # TARGET – kept in CSV but not a feature
        "flood_risk_enc",     # numeric target – kept but not a feature
    ]

    present = [c for c in non_feature_cols if c in df.columns]
    absent  = [c for c in non_feature_cols if c not in df.columns]

    if absent:
        print(f"  (Columns not found in data, nothing to drop: {absent})")

    print(f"  Non-feature columns kept in CSV but excluded from X: {present}")
    print("  NOTE: When you build your model, load flood_risk as y and")
    print("  drop all of these from X. See the FEATURE_COLS list below.")
    return df


def build_feature_list(df: pd.DataFrame) -> list[str]:
    """
    Return the list of columns that should be used as model input features (X).
    This is everything except metadata and the target label.
    """
    exclude = {
        "observation_id", "timestamp", "barangay", "municipality",
        "flood_risk", "flood_risk_enc",
        # Keep original text categoricals for human readability, but don't use
        # them as features (use the _enc versions instead)
        "flood_hazard_level", "drainage_condition",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned, feature-engineered DataFrame to a CSV file."""
    banner("STEP 8 – Save processed data")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} rows × {len(df.columns)} columns → {output_path}")


def save_quality_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    rows_before: int,
    rows_after_dedup: int,
    imputation_log: dict,
    validation_log: dict,
    output_path: Path,
) -> None:
    """
    Write a plain-text data-quality report so you can audit what the
    preprocessing pipeline did and catch any surprises before model training.

    This report is your audit trail:
    - It shows exactly how many rows were removed and why.
    - It records every fill value used for missing data.
    - It lists the final feature columns the model will see.
    Good practice is to commit this report to your Git repo alongside the code.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 70,
        "  AquaAlert – Data Quality Report",
        f"  Generated : {now}",
        f"  Input     : {RAW_FILE}",
        f"  Output    : {OUTPUT_FILE}",
        "=" * 70,
        "",
        "PROTOTYPE NOTICE",
        "  This report documents a student capstone prototype pipeline.",
        "  It does not constitute an official flood data product.",
        "",
        "─" * 70,
        "1. ROW COUNTS",
        "─" * 70,
        f"  Rows in raw file                 : {rows_before:,}",
        f"  Rows after removing duplicates   : {rows_after_dedup:,}",
        f"  Rows removed (duplicates)        : {rows_before - rows_after_dedup:,}",
        f"  Rows in final processed file     : {len(df):,}",
        f"  Total rows removed overall       : {rows_before - len(df):,}",
        "",
        "─" * 70,
        "2. MISSING-VALUE HANDLING",
        "─" * 70,
    ]

    for col, info in imputation_log.items():
        if info.get("missing_col"):
            lines.append(f"  {col:<40} column not found in input")
        elif info.get("rows_dropped", 0) > 0:
            lines.append(f"  {col:<40} {info['rows_dropped']} rows dropped "
                         f"(missing target)")
        elif info.get("n_missing", 0) > 0:
            strategy = info.get("strategy", "n/a")
            fill_val = info.get("fill_value", info.get("strategy", ""))
            lines.append(f"  {col:<40} {info['n_missing']} missing → "
                         f"strategy={strategy}, fill={fill_val}")
        else:
            lines.append(f"  {col:<40} no missing values")

    lines += [
        "",
        "─" * 70,
        "3. CATEGORICAL VALIDATION",
        "─" * 70,
    ]
    for col, info in validation_log.items():
        if info.get("missing_col"):
            lines.append(f"  {col:<30} column not found")
        else:
            n_inv = info.get("n_invalid_rows", 0)
            lines.append(f"  {col:<30} invalid rows dropped: {n_inv}")

    lines += [
        "",
        "─" * 70,
        "4. FINAL FEATURE COLUMNS  (use these as X in model training)",
        "─" * 70,
    ]
    for i, col in enumerate(feature_cols, 1):
        lines.append(f"  {i:>3}. {col}")

    lines += [
        "",
        "─" * 70,
        "5. TARGET COLUMN",
        "─" * 70,
        "  Text  : flood_risk      (Low / Moderate / High)",
        "  Numeric: flood_risk_enc (0=Low, 1=Moderate, 2=High)",
        "",
        "─" * 70,
        "6. CLASS DISTRIBUTION (flood_risk)",
        "─" * 70,
    ]
    if "flood_risk" in df.columns:
        counts = df["flood_risk"].value_counts()
        total  = len(df)
        for label in VALID_RISK_LABELS:
            n = counts.get(label, 0)
            pct = 100 * n / total if total > 0 else 0
            lines.append(f"  {label:<12} {n:>6,} rows  ({pct:.1f}%)")

    lines += [
        "",
        "─" * 70,
        "7. NOTES FOR THE TEAM",
        "─" * 70,
        textwrap.fill(
            "All numeric thresholds used in feature engineering are PROTOTYPE "
            "ASSUMPTIONS. Validate them against local NCR historical data before "
            "using this dataset in a production system.",
            width=68, initial_indent="  ", subsequent_indent="  "
        ),
        textwrap.fill(
            "The flood_risk target was labeled using proxy sources "
            "(NDRRMC sitreps, news archives, LGU bulletins). "
            "Treat label accuracy as approximate and document your annotation "
            "log alongside this report.",
            width=68, initial_indent="  ", subsequent_indent="  "
        ),
        "",
        "=" * 70,
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Data quality report saved → {output_path}")


def print_final_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    rows_before: int,
    rows_after_dedup: int,
) -> None:
    """Print a concise summary to the console at the end of the run."""
    banner("FINAL SUMMARY")
    print(f"  Raw rows loaded              : {rows_before:,}")
    print(f"  Rows after deduplication     : {rows_after_dedup:,}")
    print(f"  Final processed rows         : {len(df):,}")
    print(f"  Rows removed (total)         : {rows_before - len(df):,}")
    print(f"  Final columns in output CSV  : {len(df.columns)}")
    print(f"  Feature columns for model    : {len(feature_cols)}")
    print()
    print("  Feature columns (X for model training):")
    for col in feature_cols:
        print(f"    • {col}")
    print()
    print("  Target column  : flood_risk  (text)")
    print("  Numeric target : flood_risk_enc  (0=Low, 1=Moderate, 2=High)")
    print()
    print(f"  Output saved to : {OUTPUT_FILE}")
    print(f"  Report saved to : {REPORT_FILE}")
    print()
    print("  PROTOTYPE NOTICE: This output is for a student capstone project.")
    print("  It is NOT an official flood data product.")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Run the full preprocessing pipeline in order.
    Each step is a separate function so it is easy to read, test, and debug.
    """
    print("\n" + "=" * 65)
    print("  AquaAlert – Flood-Risk Dataset Preprocessing Pipeline")
    print("=" * 65)

    # Record how many rows we start with
    df = load_raw_data(RAW_FILE)
    rows_before = len(df)

    # Parse and extract date/time features
    df = parse_timestamps(df)

    # Remove exact duplicate rows
    df, _ = remove_duplicates(df)
    rows_after_dedup = len(df)

    # Validate and drop rows with unrecognised category values
    df, validation_log = validate_and_fix_categories(df)

    # Fill in missing values (must come AFTER validation so we don't fill
    # rows that are about to be dropped anyway)
    df, imputation_log = handle_missing_values(df)

    # Convert text categories to ordered numbers
    df = encode_categoricals(df)

    # Create new features from existing ones
    df = engineer_features(df)

    # Identify which columns are real features vs. metadata / target
    feature_cols = build_feature_list(df)
    drop_leakage_columns(df)   # prints a warning; no actual drop from the CSV

    # Save processed data and quality report
    save_output(df, OUTPUT_FILE)
    save_quality_report(
        df, feature_cols,
        rows_before, rows_after_dedup,
        imputation_log, validation_log,
        REPORT_FILE,
    )

    # Print the final console summary
    print_final_summary(df, feature_cols, rows_before, rows_after_dedup)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
