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

INPUT  : data/raw/aquaalert_raw.csv   (built by build_real_dataset.py)
OUTPUT : data/processed/aquaalert_features.csv
REPORT : reports/data_quality_report.txt

EXPECTED COLUMNS IN THE RAW FILE
---------------------------------
  observation_id               – unique row ID (e.g. "OBS-00001")
  timestamp                    – observation date (daily data)
  municipality                 – city (Manila / Quezon City / Marikina / Pasig)
  rainfall_last_24h_mm         – rainfall on the observation day (mm)
  rainfall_prev_day_mm         – rainfall on the previous day
  rainfall_last_3d_mm          – 3-day rolling rainfall total
  rainfall_last_7d_mm          – 7-day rolling rainfall total
  river_water_level_m          – river water level (m)
  water_level_change_1d_m      – change in river level since the previous day
  soil_moisture_pct            – soil moisture (%)
  elevation_m                  – city elevation (m)
  aegis_points                 – number of AEGIS points summarised for the city
  aegis_mean_flood_height_code – mean AEGIS flood-height code for the city
  aegis_share_deep_flood       – share of the city's AEGIS points with code ≥ 3
  aegis_mean_elevation_m       – mean elevation of the city's AEGIS points
  flood_hazard_level           – AEGIS-derived hazard zone: Low / Medium / High
  flood_occurrence             – real binary flood label (NOT a feature)
  flood_risk                   – TARGET LABEL: Low / Moderate / High

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

# Numeric columns that MUST have a value (no NaN allowed after imputation).
# build_real_dataset.py already drops rows without full history, so the real
# data has no gaps; median fill is only a fallback for future data.
# NOTE: the median is computed over ALL rows (train + test).  If you ever
# feed data with gaps, move imputation into the model pipeline instead.
REQUIRED_NUMERIC_COLS = [
    "rainfall_last_24h_mm",
    "rainfall_prev_day_mm",
    "rainfall_last_3d_mm",
    "rainfall_last_7d_mm",
    "river_water_level_m",
    "soil_moisture_pct",
    "elevation_m",
]

# The day-over-day level change has no value on each city's first day.
# We fill it with a special sentinel value (−999) and add a binary
# "was_missing" flag so the model can tell "no reading" from "no change".
SENSOR_COLS_NULLABLE = ["water_level_change_1d_m"]

# Valid category sets – used for validation and encoding
VALID_HAZARD_LEVELS = ["Low", "Medium", "High"]     # ordered Low → High
VALID_RISK_LABELS   = ["Low", "Moderate", "High"]   # ordered Low → High

# The sentinel value written when a reading is absent.
# Using −999 (not 0 or NaN) so the model can distinguish "no reading" from
# "reading is zero".  This is a PROTOTYPE ASSUMPTION.
MISSING_SENSOR_SENTINEL = -999.0

# Thresholds for the engineered "water_level_rise_cat" feature (metres per
# day).  PROTOTYPE ASSUMPTIONS – validate against EFCOS gauge data.
RISE_THRESHOLD_RAPID_M_PER_D      = 1.0
RISE_THRESHOLD_VERY_RAPID_M_PER_D = 2.0

# Daily rainfall used as the "1.0" point when scaling rain to [0, 1].
# PROTOTYPE ASSUMPTION (≈ the wettest day in the 2016–2020 source data).
RAIN_24H_SCALE_MM = 50.0

# Metro Manila floods "mainly during May to November" – JICA/DPWH (2014),
# Data Collection Survey on Flood Management in Metro Manila, Section 2.3.
FLOOD_SEASON_MONTHS = [5, 6, 7, 8, 9, 10, 11]

# Columns kept in the CSV for reference/target but never used as model input.
NON_FEATURE_COLS = [
    "observation_id",     # row ID
    "timestamp",          # replaced by obs_month / obs_dayofweek
    "municipality",       # location text; the aegis_* / elevation columns
                          # already carry the city's information as numbers
    "flood_risk",         # TARGET
    "flood_risk_enc",     # numeric target
    "flood_occurrence",   # source binary label the target is built from
    "flood_hazard_level", # text version; the _enc column is the feature
    "aegis_points",       # sample size of the AEGIS summary, not a measurement
]

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

    We tell pandas to treat 'observation_id' as a plain string (not a number)
    so IDs like 'OBS-00001' are preserved exactly.
    """
    banner("STEP 1 – Load raw data")

    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        print("  Run build_real_dataset.py first to create it.")
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
      sort and split by time.
    - Extracting month and day-of-week helps the model learn seasonal
      patterns (JICA 2014: Metro Manila floods mainly May–November).
      The source data is daily, so there is no hour-of-day feature.
    """
    banner("STEP 2 – Parse timestamps")

    if "timestamp" not in df.columns:
        print("  [WARNING] No 'timestamp' column found – skipping this step.")
        return df

    # 'errors=coerce' turns unparseable values to NaT instead of crashing.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    n_bad = df["timestamp"].isna().sum()
    if n_bad > 0:
        print(f"  [WARNING] {n_bad} rows had unparseable timestamps → set to NaT")

    df["obs_month"]       = df["timestamp"].dt.month         # 1–12
    df["obs_dayofweek"]   = df["timestamp"].dt.day_of_week   # 0=Monday, 6=Sunday
    df["is_flood_season"] = df["obs_month"].isin(
        FLOOD_SEASON_MONTHS
    ).astype(int)  # 1 during the May–Nov flood season (JICA 2014)

    print(f"  Timestamp range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print("  Added columns: obs_month, obs_dayofweek, is_flood_season")
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows that are exact duplicates.

    Why?
    - Duplicate rows inflate the training set without adding new information,
      and the same row might appear in both training and test sets.

    We keep the FIRST occurrence and drop the rest.
    """
    banner("STEP 3 – Remove duplicates")

    rows_before = len(df)
    n_full_dupes = df.duplicated().sum()

    # Same city on the same date twice would be a data-entry error.
    n_key_dupes = 0
    if {"timestamp", "municipality"} <= set(df.columns):
        n_key_dupes = df.duplicated(subset=["timestamp", "municipality"]).sum()

    df = df.drop_duplicates().reset_index(drop=True)

    rows_removed = rows_before - len(df)
    print(f"  Fully identical duplicate rows removed : {n_full_dupes}")
    print(f"  Duplicate (date, city) rows found      : {n_key_dupes}")
    print(f"  Rows remaining after deduplication     : {len(df):,}")
    return df, rows_removed


def validate_and_fix_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Check that categorical columns contain only expected values.
    Rows with invalid categories are dropped and counted.

    Why validate categories?
    - If someone accidentally typed "high" instead of "High", the encoder will
      treat it as an unknown category and silently produce wrong numbers.
    """
    banner("STEP 4a – Validate categorical columns")

    validation_log = {}
    rows_to_drop = pd.Series(False, index=df.index)  # start: no rows flagged

    checks = {
        "flood_hazard_level": VALID_HAZARD_LEVELS,
        "flood_risk":         VALID_RISK_LABELS,  # the target label
        "flood_occurrence":   [0, 1],             # source binary label
    }

    for col, valid_values in checks.items():
        if col not in df.columns:
            print(f"  [WARNING] Column '{col}' not found – skipping validation.")
            validation_log[col] = {"missing_col": True}
            continue

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
    Water-level change column → fill with −999 sentinel + add a binary flag column
    Categorical columns       → fill with the column MODE (most common value)
    Target label (flood_risk) → rows with missing target are DROPPED
    ─────────────────────────────────────────────────────────────────────────
    The real data has no gaps (rows without full history are dropped in
    build_real_dataset.py); this step is a safeguard for future data.
    """
    banner("STEP 4b – Handle missing values")

    imputation_log = {}

    if "flood_risk" in df.columns:
        n_missing_target = df["flood_risk"].isna().sum()
        if n_missing_target > 0:
            print(f"  Dropping {n_missing_target} rows with missing flood_risk "
                  f"(cannot train without a label)")
            df = df[df["flood_risk"].notna()].reset_index(drop=True)
        imputation_log["flood_risk"] = {"rows_dropped": int(n_missing_target)}

    # --- Level-change column: sentinel fill + missing indicator ---
    for col in SENSOR_COLS_NULLABLE:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            # No gaps → no flag column (an all-zero column carries no signal).
            imputation_log[col] = {"n_missing": 0}
            continue
        flag_col  = f"{col}_missing"

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
    for col in ["flood_hazard_level", "municipality"]:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
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

    We use ORDINAL encoding (not one-hot encoding) because the hazard level
    has a natural order:
      flood_hazard_level : Low < Medium < High  → encoded as 0, 1, 2

    The original text column is kept alongside the encoded one so you can
    still read the data easily.
    """
    banner("STEP 5 – Encode categorical features")

    encoder = OrdinalEncoder(
        categories=[VALID_HAZARD_LEVELS],
        handle_unknown="use_encoded_value",
        unknown_value=-1,                # unknown category → −1 (safe fallback)
    )

    if "flood_hazard_level" in df.columns:
        df["flood_hazard_level_enc"] = (
            encoder.fit_transform(df[["flood_hazard_level"]])[:, 0].astype(int)
        )
        print(f"  'flood_hazard_level' → 'flood_hazard_level_enc': "
              f"{dict(zip(encoder.categories_[0], range(len(encoder.categories_[0]))))}")

    # Encode the TARGET label separately using a simple map so we keep control.
    if "flood_risk" in df.columns:
        risk_map = {"Low": 0, "Moderate": 1, "High": 2}
        df["flood_risk_enc"] = df["flood_risk"].map(risk_map)
        print(f"  'flood_risk' → 'flood_risk_enc': {risk_map}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns (features) that combine or transform existing ones.

    Features created here:
    ──────────────────────────────────────────────────────────────────────────
    1. rainfall_ratio_1d_7d
       Today's rainfall divided by the 7-day total (clipped to [0, 1]).
       Why: a high ratio means most of the week's rain fell today – a sudden
       burst that overwhelms drainage faster than steady rain of the same total.

    2. water_level_rise_cat
       Classifies the day-over-day river change into:
         0 = No previous reading / falling / stable
         1 = Rising (slow)
         2 = Rising rapidly      (≥ 1.0 m/day – PROTOTYPE ASSUMPTION)
         3 = Rising very rapidly (≥ 2.0 m/day – PROTOTYPE ASSUMPTION)

    3. combined_rain_hazard_risk
       Scaled daily rainfall (0–1) × AEGIS hazard level (0–1).
       Why: the same rain is more dangerous in a city whose AEGIS points show
       deep historical flooding (e.g. Marikina) than in a low-hazard city.

    4. wet_antecedent_index
       Soil moisture (0–1) × scaled 3-day rainfall (0–1).
       Why: rain falling on already-saturated ground runs off instead of
       soaking in.

    TARGET LEAKAGE CHECK
    ────────────────────
    None of these features use 'flood_risk' or 'flood_occurrence'.
    They are all derived from input observations only.
    """
    banner("STEP 6 – Engineer new features")

    if {"rainfall_last_24h_mm", "rainfall_last_7d_mm"} <= set(df.columns):
        df["rainfall_ratio_1d_7d"] = (
            df["rainfall_last_24h_mm"]
            / df["rainfall_last_7d_mm"].replace(0, np.nan)  # 0 → NaN to avoid /0
        ).fillna(0).clip(0, 1)
        print("  + rainfall_ratio_1d_7d       (0=steady rain, 1=all in one day)")

    if "water_level_change_1d_m" in df.columns:
        def classify_rise(change: float) -> int:
            """Sentinel −999 means no previous reading; those rows get 0."""
            if change == MISSING_SENSOR_SENTINEL or change <= 0:
                return 0    # No previous reading, falling, or stable
            elif change >= RISE_THRESHOLD_VERY_RAPID_M_PER_D:
                return 3
            elif change >= RISE_THRESHOLD_RAPID_M_PER_D:
                return 2
            else:
                return 1

        df["water_level_rise_cat"] = df["water_level_change_1d_m"].apply(classify_rise)
        print("  + water_level_rise_cat       (0=stable/no data, 1=slow, "
              "2=rapid, 3=very rapid)")

    if {"rainfall_last_24h_mm", "flood_hazard_level_enc"} <= set(df.columns):
        rain_scaled   = (df["rainfall_last_24h_mm"] / RAIN_24H_SCALE_MM).clip(0, 1)
        hazard_scaled = df["flood_hazard_level_enc"] / 2.0
        df["combined_rain_hazard_risk"] = rain_scaled * hazard_scaled
        print("  + combined_rain_hazard_risk  (0=no risk, 1=extreme rain in high-hazard city)")

    if {"soil_moisture_pct", "rainfall_last_3d_mm"} <= set(df.columns):
        rain3_scaled = (df["rainfall_last_3d_mm"] / (3 * RAIN_24H_SCALE_MM)).clip(0, 1)
        df["wet_antecedent_index"] = (df["soil_moisture_pct"] / 100.0) * rain3_scaled
        print("  + wet_antecedent_index       (saturated ground × recent rain)")

    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report the columns that must NOT be used as model input features.

    TARGET LEAKAGE means accidentally including information that directly
    reveals the answer during training.  Leaky columns here:
    - flood_risk / flood_risk_enc (the label)
    - flood_occurrence (the source binary label the 3-class target is built on)

    These are kept in the output CSV; the training scripts exclude them from X.
    """
    banner("STEP 7 – Drop non-feature / leakage columns")

    present = [c for c in NON_FEATURE_COLS if c in df.columns]
    absent  = [c for c in NON_FEATURE_COLS if c not in df.columns]

    if absent:
        print(f"  (Columns not found in data, nothing to drop: {absent})")

    print(f"  Non-feature columns kept in CSV but excluded from X: {present}")
    print("  NOTE: When you build your model, load flood_risk as y and")
    print("  drop all of these from X.")
    return df


def build_feature_list(df: pd.DataFrame) -> list[str]:
    """Return the columns that should be used as model input features (X)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


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
        "DATA SOURCES (see build_real_dataset.py)",
        "  1. Flood Prediction NCR Philippines – daily, 2016–2020,",
        "     Manila / Quezon City / Marikina / Pasig (time series + label)",
        "  2. AEGIS flood points – static per-city flood-hazard summary",
        "  3. JICA/DPWH (2014) Data Collection Survey on Flood Management",
        "     in Metro Manila – flood season (May–Nov), major-flood table",
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
            fill_val = info.get("fill_value")
            fill_txt = f", fill={fill_val}" if fill_val is not None else ""
            lines.append(f"  {col:<40} {info['n_missing']} missing → "
                         f"strategy={strategy}{fill_txt}")
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
            "flood_risk is derived from the source binary FloodOccurrence: "
            "High = recorded flood; Moderate = no flood but daily rain >= 20 mm "
            "or river level >= its training-period 90th percentile (2.4 m); Low = otherwise. In the source data "
            "FloodOccurrence equals (rainfall >= 30 mm) on every row, so the "
            "target is a deterministic function of rainfall and water level.",
            width=68, initial_indent="  ", subsequent_indent="  "
        ),
    ]
    if "rainfall_last_24h_mm" in df.columns:
        lines.append(textwrap.fill(
            f"The wettest day in the data is "
            f"{df['rainfall_last_24h_mm'].max():.1f} mm. Documented major Metro "
            "Manila floods (JICA 2014, Table 2.6) had 149-291 mm of basin "
            "rainfall in one day, so typhoon-scale events are absent from "
            "training.",
            width=68, initial_indent="  ", subsequent_indent="  "
        ))
    lines += ["", "=" * 70]

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
    """Run the full preprocessing pipeline in order."""
    print("\n" + "=" * 65)
    print("  AquaAlert – Flood-Risk Dataset Preprocessing Pipeline")
    print("=" * 65)

    df = load_raw_data(RAW_FILE)
    rows_before = len(df)

    df = parse_timestamps(df)

    df, _ = remove_duplicates(df)
    rows_after_dedup = len(df)

    df, validation_log = validate_and_fix_categories(df)

    # Fill missing values AFTER validation so we don't fill rows that are
    # about to be dropped anyway
    df, imputation_log = handle_missing_values(df)

    df = encode_categoricals(df)

    df = engineer_features(df)

    feature_cols = build_feature_list(df)
    drop_leakage_columns(df)   # reports only; no actual drop from the CSV

    save_output(df, OUTPUT_FILE)
    save_quality_report(
        df, feature_cols,
        rows_before, rows_after_dedup,
        imputation_log, validation_log,
        REPORT_FILE,
    )

    print_final_summary(df, feature_cols, rows_before, rows_after_dedup)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
