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

**AquaAlert** is a machine-learning system that classifies flood risk at the barangay level into three categories: **Low**, **Moderate**, and **High**. It is designed to demonstrate how sensor data, hazard maps, and community reports could be combined into a single decision-support score for local governments and residents in Metro Manila (NCR), Philippines.

Two classifiers are built and compared:

| Model | Role | Key strength |
|---|---|---|
| Logistic Regression | Baseline | Highly interpretable; coefficients explain each feature's contribution |
| Random Forest | Main model | Handles non-linear interactions; includes hyperparameter tuning via GridSearchCV |

The system was built as an undergraduate capstone project and is trained on **synthetic (computer-generated) data**. It does **not** issue official flood warnings and does **not** replace PAGASA, LGU DRRMOs, or the NDRRMC.

---

## 2. Real-World Problem

The Philippines is among the world's most flood-prone countries. Metro Manila alone experiences severe flooding during the June–November typhoon season, affecting millions of residents annually. Key challenges include:

- **Uneven warning reach** — official advisories may not reach individual barangays fast enough.
- **Information fragmentation** — rainfall data, river levels, drainage conditions, and community reports exist in separate systems.
- **Reactive responses** — many residents receive warnings only after floodwater has already begun rising.

AquaAlert explores whether a lightweight machine-learning model, combining publicly available sensor data and crowdsourced community reports, can provide an **earlier, barangay-level risk signal** to complement official warnings.

---

## 3. Project Objectives

1. Build a reproducible ML pipeline that classifies barangay flood risk from five sensor and community features.
2. Compare a Logistic Regression baseline against a Random Forest main model on the same held-out test set.
3. Demonstrate that **High-risk recall** (catching real flood events) is the most safety-critical metric — more important than accuracy alone.
4. Document all assumptions, limitations, and responsible-use guidelines transparently.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  generate_sample_raw.py                                     │
│  → Synthetic raw CSV (300 rows, 8 NCR barangays)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                        │
│  preprocess.py                                              │
│  → Clean + validate → Engineer 5 features                   │
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

**Features used by both models:**

| Feature | Description |
|---|---|
| `rainfall_last_3h_mm` | Cumulative rainfall in the past 3 hours (mm) |
| `river_water_level_m` | Current river water level at the nearest gauge (m) |
| `flood_hazard_level_enc` | NAMRIA/LGU static flood-hazard zone (0 = Low, 1 = Medium, 2 = High) |
| `drainage_condition_enc` | Observed drainage blockage (0 = Clean, 1 = Partly Blocked, 2 = Blocked) |
| `citizen_report_count` | Number of community flood reports in the past 3 hours |

All thresholds used to engineer categorical features are assumptions and must be validated against local expert knowledge before any operational use.

---

## 5. Repository Folder Structure

```
aquaalert/
│
├── README.md                        ← you are here
│
├── generate_sample_raw.py           ← Step 0: create synthetic raw data
├── preprocess.py                    ← Step 1: clean + feature engineering
├── train_baseline_lr.py             ← Step 2: Logistic Regression baseline
├── train_random_forest.py           ← Step 3: Random Forest main model
├── compare_models.py                ← Step 4: head-to-head comparison
│
├── data/
│   ├── raw/
│   │   └── aquaalert_raw.csv        ← generated by generate_sample_raw.py
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
| `timestamp` | datetime | ISO 8601 observation time |
| `barangay` | string | Barangay name |
| `municipality` | string | Municipality / city |
| `rainfall_last_3h_mm` | float | 3-hour cumulative rainfall (mm) |
| `rainfall_last_24h_mm` | float | 24-hour cumulative rainfall (mm) |
| `river_water_level_m` | float | River water level (m); `-999.0` = sensor offline |
| `elevation_m` | float | Barangay mean elevation (m) |
| `flood_hazard_level` | string | Static hazard zone: Low / Medium / High |
| `drainage_condition` | string | Observed drainage: Clean / Partly Blocked / Blocked |
| `citizen_report_count` | int | Community reports in past 3 hours |
| `flood_risk` | string | Target label: Low / Moderate / High |

### Data source attribution

> **⚠ Important:** The current dataset is entirely **synthetic** — generated by `generate_sample_raw.py` to simulate realistic NCR flood patterns for academic purposes. It is **not** derived from real sensor readings.

For a production system, the following real data sources would replace the synthetic data:

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

### 8a. Generate synthetic raw data

```bash
python generate_sample_raw.py
```

This creates `data/raw/aquaalert_raw.csv` — a 300-row synthetic dataset with 8 NCR barangays. The script intentionally introduces 5 duplicate rows, 8 missing values, and 3 invalid category labels to simulate real-world data quality issues.

### 8b. Run preprocessing

```bash
python preprocess.py
```

This script:
- Removes duplicates and fills or flags missing sensor values
- Validates categorical columns (`flood_hazard_level`, `drainage_condition`, `flood_risk`)
- Encodes categories as integers for the model
- Engineers 5 additional features (rainfall intensity category, rainfall ratio, water-level rise rate, report density, combined rain-drainage risk)
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

- **Do not choose a model based on accuracy alone.** The dataset is class-imbalanced (~6% High-risk rows), so accuracy can be high even when the model misses every flood event.
- Prioritise **High-risk recall** first. A false negative (missed High event) could leave a community without warning.
- Check **Moderate-risk recall** second.
- Use macro F1 to compare overall balance across all three classes.

---

## 12. Expected Output Files

After running all four scripts in order, the following files should exist:

```
data/
  raw/aquaalert_raw.csv                    ✓ after generate_sample_raw.py
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

**1. Synthetic training data.**
All 300 rows were generated by a script using probability weights — not collected from real sensors or real flood events. The model learned patterns in artificial data. Its performance on real NCR observations is unknown and may be substantially different.

**2. Very small High-risk sample.**
Approximately 18 rows in the full dataset are labelled High risk. With so few examples, a single misclassification shifts High-recall by ~0.06. All High-class metrics carry wide uncertainty intervals and should not be treated as reliable estimates.

**3. No real-time sensor integration.**
The pipeline reads a static CSV. There is no live connection to PAGASA AWS stations, EFCOS river gauges, or any other data feed.

**4. Manually assessed drainage condition.**
The `drainage_condition` feature assumes a human observer reports blockage. No automated drain sensor is modelled.

**5. Static hazard zones.**
`flood_hazard_level_enc` uses a fixed hazard map. It does not update with changing land use, new infrastructure, or updated LiDAR surveys.

**6. No spatial context.**
Latitude, longitude, distance to river channels, and proximity to other flood-prone areas are not features in the current model.

**7. No temporal autocorrelation.**
Flood risk at time T is correlated with risk at T−1. Neither model accounts for this. A time-series approach (LSTM, LightGBM with lag features) may be more appropriate with real sequential data.

**8. All thresholds are assumptions.**
Every numeric threshold in the preprocessing and scoring code (e.g., 50 mm/3h = Moderate rainfall intensity, 100 mm/3h = High) was chosen for demonstration purposes and has not been validated against PAGASA warning criteria or expert domain knowledge.

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
If you reference this project in a paper or report, please state clearly that results are from a synthetic-data project and do not represent validated operational performance.

---

## 16. Contributors

**Tools and libraries used:**
- [scikit-learn](https://scikit-learn.org/) — machine learning
- [imbalanced-learn](https://imbalanced-learn.org/) — SMOTE oversampling
- [pandas](https://pandas.pydata.org/) — data manipulation
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) — visualisation
- [joblib](https://joblib.readthedocs.io/) — model serialisation

---

*AquaAlert — College Capstone Project · NCR, Philippines · 2026*
*This system does not replace and must not be mistaken for any official government flood warning service.*
