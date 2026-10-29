# AquaAlert 

> **Hyper-local flood-risk decision support for NCR barangays**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Real-World Problem](#2-real-world-problem)
3. [Project Objectives](#3-project-objectives)
4. [System Architecture](#4-system-architecture)
5. [Repository Folder Structure](#5-repository-folder-structure)
6. [Dataset Schema and Data Sources](#6-dataset-schema-and-data-sources)
7. [Python Environment Setup](#7-python-environment-setup)
8. [Step 1 — Prepare the Data](#8-step-1--prepare-the-data)
9. [Step 2 — Train the Logistic Regression Baseline](#9-step-2--train-the-logistic-regression-baseline)
10. [Step 3 — Train the Random Forest Main Model](#10-step-3--train-the-random-forest-main-model)
11. [Step 4 — Compare Both Models](#11-step-4--compare-both-models)
12. [Expected Output Files](#12-expected-output-files)
13. [Reproducibility Notes](#13-reproducibility-notes)
14. [Limitations](#14-limitations)
15. [Responsible-Use Disclaimer](#15-responsible-use-disclaimer)
16. [Contributors](#16-contributors)

---

## 1. Project Description

**AquaAlert** is a machine-learning system that classifies flood risk (currently at the city level, with barangay level as the goal) into three categories: **Low**, **Moderate**, and **High**. It is designed to demonstrate how sensor data, hazard maps, and community reports could be combined into a single decision-support score for local governments and residents in Metro Manila (NCR), Philippines.

Two classifiers are built and compared:

| Model | Role | Key strength |
|---|---|---|
| Logistic Regression | Baseline | Highly interpretable; coefficients explain each feature's contribution |
| Random Forest | Main model | Handles non-linear interactions; includes hyperparameter tuning via GridSearchCV |

The system was built as an undergraduate capstone project and is trained on **real public data**: daily 2016–2020 flood observations for Manila, Quezon City, Marikina and Pasig, a per-city hazard map built from the AEGIS flood-height dataset, and context from the JICA/DPWH (2014) Metro Manila flood survey (see [Section 6](#6-dataset-schema-and-data-sources)). It does **not** issue official flood warnings and does **not** replace PAGASA, LGU DRRMOs, or the NDRRMC.

---

## 2. Real-World Problem

The Philippines is among the world's most flood-prone countries. Metro Manila alone experiences severe flooding during the June–November typhoon season, affecting millions of residents annually. Key challenges include:

- **Uneven warning reach** — official advisories may not reach individual barangays fast enough.
- **Information fragmentation** — rainfall data, river levels, drainage conditions, and community reports exist in separate systems.
- **Reactive responses** — many residents receive warnings only after floodwater has already begun rising.

AquaAlert explores whether a lightweight machine-learning model, combining publicly available sensor data and crowdsourced community reports, can provide an **earlier, barangay-level risk signal** to complement official warnings.

---

## 3. Project Objectives

1. Build a reproducible ML pipeline that classifies daily city-level flood risk from rainfall, river-level, soil-moisture and hazard-map features.
2. Compare a Logistic Regression baseline against a Random Forest main model on the same held-out test set.
3. Demonstrate that **High-risk recall** (catching real flood events) is the most safety-critical metric — more important than accuracy alone.
4. Document all assumptions, limitations, and responsible-use guidelines transparently.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  build_real_dataset.py                                      │
│  → NCR daily series + AEGIS hazard + JICA context           │
│  → aquaalert_raw.csv (7,308 rows, 4 NCR cities, 2016–2020)  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                        │
│  preprocess.py                                              │
│  → Clean + validate → Engineer 4 features                   │
│  → aquaalert_features.csv + data_quality_report.txt         │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────────────────────────┐
│  BASELINE MODEL  │   │           MAIN MODEL                 │
│  train_baseline  │   │  train_random_forest.py              │
│  _lr.py          │   │  ImbPipeline: SMOTE → Scaler → RF    │
│  StandardScaler  │   │  GridSearchCV (5-fold, f1_macro)     │
│  + LogisticReg   │   │                                      │
└────────┬─────────┘   └──────────────────┬───────────────────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  COMPARISON LAYER                           │
│  compare_models.py                                          │
│  → Same test set → CSV table, bar chart, confusion matrices │
│  → Auto-generated interpretation report                     │
└─────────────────────────────────────────────────────────────┘
```

**Main features used by both models** (20 in total; the full list is in `reports/data_quality_report.txt`):

| Feature | Description |
|---|---|
| `rainfall_last_24h_mm` | Rainfall on the observation day (mm) |
| `rainfall_prev_day_mm`, `rainfall_last_3d_mm`, `rainfall_last_7d_mm` | Previous-day and rolling 3-/7-day rainfall (past values only) |
| `river_water_level_m`, `water_level_change_1d_m` | River water level (m) and its change since the previous day |
| `soil_moisture_pct` | Soil moisture (%) |
| `flood_hazard_level_enc` | Static hazard zone built from AEGIS flood heights (0 = Low, 1 = Medium, 2 = High) |
| `aegis_mean_flood_height_code`, `aegis_share_deep_flood` | Per-city AEGIS flood-height summaries |
| `is_flood_season` | 1 in May–November, the Metro Manila flood season per JICA/DPWH (2014) |
| `rainfall_ratio_1d_7d`, `water_level_rise_cat`, `combined_rain_hazard_risk`, `wet_antecedent_index` | Engineered features (see `preprocess.py`) |

All thresholds used to engineer categorical features are assumptions and must be validated against local expert knowledge before any operational use.

---

## 5. Repository Folder Structure

```
aquaalert/
│
├── README.md                        ← you are here
│
├── build_real_dataset.py            ← Step 0: build raw CSV from real sources
├── preprocess.py                    ← Step 1: clean + feature engineering
├── train_baseline_lr.py             ← Step 2: Logistic Regression baseline
├── train_random_forest.py           ← Step 3: Random Forest main model
├── compare_models.py                ← Step 4: head-to-head comparison
│
├── data/
│   ├── raw/
│   │   ├── sources/                 ← original real datasets (see Section 6)
│   │   │   ├── flood_prediction_ncr_2016_2020.csv
│   │   │   ├── aegis_flood_points.csv
│   │   │   └── jica_2014_major_floods.csv
│   │   └── aquaalert_raw.csv        ← generated by build_real_dataset.py
│   └── processed/
│       └── aquaalert_features.csv   ← generated by preprocess.py
│
├── models/
│   ├── baseline_lr_pipeline.pkl     ← saved LR pipeline
│   └── rf_pipeline_final.pkl        ← saved RF pipeline
│
├── reports/
│   ├── data_quality_report.txt
│   ├── lr_classification_report.csv
│   ├── lr_confusion_matrix.png
│   ├── lr_feature_weights.csv
│   ├── rf_classification_report.csv
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importance.png
│   ├── rf_feature_importance.csv
│   ├── rf_cv_results.csv
│   ├── rf_vs_baseline_summary.txt
│   ├── comparison_metrics.csv
│   ├── comparison_bar_chart.png
│   ├── comparison_confusion_matrices.png
│   └── comparison_interpretation.txt
│
└── requirements.txt
```

---

## 6. Dataset Schema and Data Sources

### Raw CSV schema (`data/raw/aquaalert_raw.csv`)

| Column | Type | Description |
|---|---|---|
| `observation_id` | string | Unique row identifier |
| `timestamp` | date | Observation date (one row per city per day) |
| `municipality` | string | Manila / Quezon City / Marikina / Pasig |
| `rainfall_last_24h_mm` | float | Rainfall on the day (mm) |
| `rainfall_prev_day_mm` | float | Rainfall on the previous day (blank on each city's first day) |
| `rainfall_last_3d_mm`, `rainfall_last_7d_mm` | float | Rolling 3-/7-day rainfall totals (blank until enough history) |
| `river_water_level_m` | float | River water level (m) |
| `water_level_change_1d_m` | float | Change in river level since the previous day |
| `soil_moisture_pct` | float | Soil moisture (%) |
| `elevation_m` | float | City elevation (m) |
| `aegis_*` | float | Per-city summaries of the AEGIS flood points |
| `flood_hazard_level` | string | AEGIS-derived hazard zone: Low / Medium / High |
| `flood_occurrence` | int | Source flood label (0/1). **Not a feature.** |
| `flood_risk` | string | Target label: Low / Moderate / High |

### Data sources

All three sources are stored unchanged (or, for the PDF, transcribed) in `data/raw/sources/`, and `build_real_dataset.py` combines them.

| File | Source | How it is used |
|---|---|---|
| `flood_prediction_ncr_2016_2020.csv` | *Flood Prediction NCR Philippines* dataset: 7,308 daily rows, 2016–2020, 4 cities, with rainfall, water level, soil moisture, elevation and a 0/1 flood flag | The time series and the flood label |
| `aegis_flood_points.csv` | *AEGIS* flood dataset: 3,510 Metro Manila points with an ordinal flood-height code (0–8), elevation and precipitation | Each point is assigned to its nearest city centre. The share of points with code ≥ 3 gives the hazard level: Marikina = High (0.59), Quezon City = Medium (0.36), Pasig = Low (0.25), Manila = Low (0.20) |
| `jica_2014_major_floods.csv` | JICA / DPWH (May 2014), *Data Collection Survey on Flood Management Plan in Metro Manila – Final Report*, Tables 2.6–2.7 | Flood season (May–Nov, Section 2.3) and a check of training rainfall against documented major floods (149–291 mm/day) |

### How the Low / Moderate / High label is built

The source data only records whether a flood happened. `build_real_dataset.py` turns that into three classes:

| Class | Rule | Rows |
|---|---|---|
| **High** | A flood was recorded (`FloodOccurrence = 1`) | 132 (1.8 %) |
| **Moderate** | No flood, but daily rain ≥ 20 mm **or** river level ≥ 2.5 m (the 90th percentile of non-flood days) | 1,022 (14.0 %) |
| **Low** | Everything else | 6,154 (84.2 %) |

> **⚠ Important:** In the source data, `FloodOccurrence` is exactly `Rainfall_mm ≥ 30` on every row. The target is therefore a fixed rule on rainfall and water level, and both models score close to perfect. This shows the models can recover those thresholds. It is **not** evidence that they can forecast real floods.

For a production system, the following data sources should be added:

| Data type | Source | Notes |
|---|---|---|
| Rainfall | [PAGASA](https://www.pagasa.dost.gov.ph/) | Automatic Weather Stations (AWS) |
| River water level | [EFCOS / DPWH](https://www.dpwh.gov.ph/) | Water level telemetry network |
| Flood hazard maps | [NAMRIA](https://www.namria.gov.ph/) | 1:10,000 scale LiDAR-derived maps |
| LGU hazard zones | Local DRRMOs | Varies per city/municipality |
| Community reports | AquaAlert app users | Crowdsourced, privacy-preserving |

---

## 7. Python Environment Setup

### Requirements

- Python **3.10 or higher**
- pip (comes with Python)
- Recommended: a virtual environment

### Install steps

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/aquaalert.git
cd aquaalert
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

> **Tip:** If you see an error about `imbalanced-learn`, run:
> `pip install imbalanced-learn --upgrade`

---

## 8. Step 1 — Prepare the Data

### 8a. Build the raw dataset from the real sources

```bash
python build_real_dataset.py
```

This creates `data/raw/aquaalert_raw.csv` (7,308 rows: 4 cities × 1,827 days). It adds past-only rainfall lags and rolling totals, the AEGIS hazard level for each city and the 3-class label. It also prints a check against the JICA major-flood table. The lag columns are blank on each city's first days, and `preprocess.py` fills them.

### 8b. Run preprocessing

```bash
python preprocess.py
```

This script:
- Removes duplicates and fills or flags missing sensor values
- Validates categorical columns (`flood_hazard_level`, `flood_risk`, `flood_occurrence`)
- Encodes categories as integers for the model
- Engineers 4 additional features (1-day/7-day rainfall ratio, water-level rise rate, combined rain-hazard risk, wet-antecedent index)
- Outputs `data/processed/aquaalert_features.csv`
- Saves a data quality summary to `reports/data_quality_report.txt`

---

## 9. Step 2 — Train the Logistic Regression Baseline

```bash
python train_baseline_lr.py
```

**What this does:**

- Sorts observations by timestamp and uses the earliest **70%** as training data, holding out the most recent **30%** as the test set.
- Fits a pipeline: `StandardScaler → LogisticRegression` with `class_weight="balanced"` to handle class imbalance.
- Evaluates on the held-out test set and prints per-class precision, recall, and F1.
- Saves all outputs to `models/` and `reports/`.

**Key metric to check after running:**

```
High-risk recall  (target ≥ 0.80)
```

A High-recall close to 1.00 means the model catches nearly every real flood event. Missing a High event is a false negative — the most dangerous kind of error for this system.

---

## 10. Step 3 — Train the Random Forest Main Model

```bash
python train_random_forest.py
```

**What this does:**

- Uses the **same 70/30 time-based split** as the baseline for a fair comparison.
- Fits an imbalanced-learn pipeline: `SMOTE → StandardScaler → RandomForestClassifier`.
  - SMOTE creates synthetic minority-class rows **inside** each cross-validation fold only — no data leakage.
- Runs a `GridSearchCV` over 27 hyperparameter combinations (3 × 3 × 3) with 5-fold stratified cross-validation, optimising for `f1_macro`.
- Evaluates on the held-out test set, prints feature importances, and generates a side-by-side comparison against the LR baseline.

**GridSearch parameters explored:**

| Parameter | Values |
|---|---|
| `n_estimators` (number of trees) | 100, 200, 300 |
| `max_depth` | None, 10, 20 |
| `min_samples_leaf` | 1, 2, 4 |

> This step takes approximately 2–5 minutes depending on your machine.

---

## 11. Step 4 — Compare Both Models

```bash
python compare_models.py
```

**Prerequisites:** Steps 1–3 must be completed first (both `.pkl` model files must exist in `models/`).

**What this produces:**

| Output file | Contents |
|---|---|
| `reports/comparison_metrics.csv` | Full metric table — one row per model |
| `reports/comparison_bar_chart.png` | Side-by-side bar chart of 6 key metrics |
| `reports/comparison_confusion_matrices.png` | Raw + normalised confusion matrices for both models |
| `reports/comparison_interpretation.txt` | Auto-generated plain-language write-up |

**How to read the comparison:**

- **Do not choose a model based on accuracy alone.** The dataset is class-imbalanced (~1.8% High-risk rows), so accuracy can be high even when the model misses every flood event.
- Prioritise **High-risk recall** first. A false negative (missed High event) could leave a community without warning.
- Check **Moderate-risk recall** second.
- Use macro F1 to compare overall balance across all three classes.

---

## 12. Expected Output Files

After running all four scripts in order, the following files should exist:

```
data/
  raw/aquaalert_raw.csv                    ✓ after build_real_dataset.py
  processed/aquaalert_features.csv         ✓ after preprocess.py

models/
  baseline_lr_pipeline.pkl                 ✓ after train_baseline_lr.py
  rf_pipeline_final.pkl                    ✓ after train_random_forest.py

reports/
  data_quality_report.txt                  ✓ after preprocess.py
  lr_classification_report.csv             ✓ after train_baseline_lr.py
  lr_confusion_matrix.png                  ✓ after train_baseline_lr.py
  lr_feature_weights.csv                   ✓ after train_baseline_lr.py
  rf_classification_report.csv             ✓ after train_random_forest.py
  rf_confusion_matrix.png                  ✓ after train_random_forest.py
  rf_feature_importance.png                ✓ after train_random_forest.py
  rf_feature_importance.csv                ✓ after train_random_forest.py
  rf_cv_results.csv                        ✓ after train_random_forest.py
  rf_vs_baseline_summary.txt               ✓ after train_random_forest.py
  comparison_metrics.csv                   ✓ after compare_models.py
  comparison_bar_chart.png                 ✓ after compare_models.py
  comparison_confusion_matrices.png        ✓ after compare_models.py
  comparison_interpretation.txt            ✓ after compare_models.py
```

---

## 13. Reproducibility Notes

All scripts use `random_state=42` throughout. Given the same Python and library versions, every run should produce **identical** outputs.

| Setting | Value |
|---|---|
| Random seed | `42` (all scripts) |
| Train/test split | 70% / 30%, time-sorted (no shuffle) |
| CV strategy | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| SMOTE k-neighbours | `3` (small minority class) |
| LR solver | `lbfgs`, `max_iter=1000`, `C=1.0` |
| RF scoring | `f1_macro` |

**Recommended library versions** (from `requirements.txt`):

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
imbalanced-learn==0.12.3
matplotlib==3.8.4
seaborn==0.13.2
joblib==1.4.2
```

To save your exact installed versions for future reproduction:

```bash
pip freeze > requirements_lock.txt
```

---

## 14. Limitations

The following limitations must be understood before interpreting any results from this project.

**1. The label follows fixed thresholds.**
The data is real, but its flood flag is exactly `rainfall ≥ 30 mm`, and the Moderate class is defined by rainfall and river-level thresholds. Test scores (RF macro F1 ≈ 0.99, LR ≈ 0.89) show how well each model recovers those rules. They are not validated flood-forecasting performance.

**2. No typhoon-scale events.**
The wettest day in 2016–2020 is 52.7 mm. The major Metro Manila floods documented by JICA/DPWH (2014) (Seniang 2000, Chedeng 2003, Winnie 2004, Ondoy 2009, Kirogi 2012) had 149–291 mm of basin rainfall in one day. The models have never seen an event like these.

**3. Small High-risk sample.**
There are 132 High-risk rows, and 50 of them are in the test set, so one misclassification moves High-recall by 0.02.

**4. City-level, daily resolution.**
There are four cities and one reading per city per day. The data has no barangay detail, no hourly rainfall bursts, no drainage condition and no citizen reports.

**5. Approximate hazard map.**
AEGIS points are assigned to cities by nearest city centre, not official boundaries. The hazard levels are fixed and relative to the four cities. The meaning of the AEGIS flood-height codes (commonly read as 0 = none, 3 = waist-high) is not documented in the file.

**6. No real-time sensor integration.**
The pipeline reads static CSVs. There is no live connection to PAGASA AWS stations, EFCOS river gauges, or any other data feed.

**7. No temporal autocorrelation.**
Flood risk at time T is correlated with risk at T−1. Neither model accounts for this. A time-series approach (LSTM, LightGBM with lag features) may be more appropriate with real sequential data.

**8. All thresholds are assumptions.**
Every numeric threshold in the dataset builder and preprocessing code (e.g., 20 mm/day or 2.5 m river level = Moderate, 1.0 m/day = rapid rise) was chosen for demonstration purposes and has not been validated against PAGASA warning criteria or expert domain knowledge.

---

## 15. Responsible-Use Disclaimer

> **⚠ Please read this section before using, sharing, or citing this project.**

AquaAlert is a **college capstone project** developed for academic evaluation. It is **not** an official flood warning system and has **not** been validated by PAGASA, NDRRMC, or any LGU DRRMO.

**Do not:**
- Use AquaAlert outputs as the sole basis for evacuation decisions.
- Present AquaAlert risk classifications to the public as official flood alerts.
- Deploy this system in a production environment without retraining on real, validated data and review by qualified disaster risk management professionals.

**Always follow official guidance from:**
- 🌦 **PAGASA** — [pagasa.dost.gov.ph](https://www.pagasa.dost.gov.ph/) — official weather and flood advisories
- 🏛 **LGU DRRMO** — your local barangay or city disaster risk reduction and management office
- 🇵🇭 **NDRRMC** — [ndrrmc.gov.ph](https://www.ndrrmc.gov.ph/) — national disaster response hotline: **911**

**Privacy (RA 10173):**
The community report feature is designed around the Philippine Data Privacy Act. No name, phone number, or exact address is collected. Photos have EXIF location data removed client-side before any upload. Reporter tokens are anonymous UUIDs generated locally.

**Citation for academic use:**
If you reference this project in a paper or report, cite the three data sources in Section 6. State clearly that the 3-class label comes from rainfall and river-level thresholds, so the results do not represent validated operational performance.

---

## 16. Contributors

**Tools and libraries used:**
- [scikit-learn](https://scikit-learn.org/) — machine learning
- [imbalanced-learn](https://imbalanced-learn.org/) — SMOTE oversampling
- [pandas](https://pandas.pydata.org/) — data manipulation
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) — visualisation
- [joblib](https://joblib.readthedocs.io/) — model serialisation

**Team**
-Francisco, Eliesha Mae L.
-Gemina, Jamaica
-Labaco, Charina Mae
-Viray, Rodelyn U.
-Viri, Aira Kristel

---

*AquaAlert — College Capstone Project · NCR, Philippines · 2026*
*This system does not replace and must not be mistaken for any official government flood warning service.*
