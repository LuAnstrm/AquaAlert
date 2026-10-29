"""
build_real_dataset.py
=====================
AquaAlert – Build the raw training CSV from REAL source datasets
-----------------------------------------------------------------
Replaces the old generate_sample_raw.py (synthetic data).

SOURCES (all stored in data/raw/sources/)
-----------------------------------------
1. flood_prediction_ncr_2016_2020.csv
     "Flood Prediction NCR Philippines" – daily observations, 2016-01-01 to
     2020-12-31, for Manila, Quezon City, Marikina and Pasig.
     Columns: Date, Location, Rainfall_mm, WaterLevel_m, SoilMoisture_pct,
              Elevation_m, FloodOccurrence (0/1)
     → provides the time series and the real flood / no-flood label.

2. aegis_flood_points.csv
     AEGIS flood dataset – 3,510 geo-referenced points in Metro Manila with an
     ordinal flood-height code (flood_heig, 0 = no flood … 8 = deepest),
     elevation and precipitation.  No dates.
     → used as a STATIC hazard map: every point is assigned to the nearest
       NCR city centre and summarised per city into a flood_hazard_level.

3. jica_2014_major_floods.csv
     Transcribed from JICA / DPWH (2014) "Data Collection Survey on Flood
     Management in Metro Manila – Final Report", Tables 2.6 and 2.7
     (major floods 2000–2012: Sto. Niño peak water level, 1-day basin
     rainfall, casualties).  The same report states that Metro Manila floods
     occur mainly from May to November (Section 2.3).
     → used for context checks (are typhoon-scale events in the training
       data?) and for the May–November flood-season feature.

HOW THE 3-CLASS LABEL IS BUILT  (PROTOTYPE ASSUMPTION – documented)
-------------------------------------------------------------------
The source dataset only says flood / no flood.  AquaAlert needs
Low / Moderate / High, so:
    High     = FloodOccurrence == 1 (a recorded flood)
    Moderate = no flood, but a near-miss: daily rain ≥ 20 mm OR river
               level ≥ 2.5 m (2.5 m = 90th percentile of non-flood days)
    Low      = everything else
NOTE: in the source data FloodOccurrence is exactly Rainfall_mm ≥ 30, so the
label is a deterministic function of rainfall and water level.  Expect very
high model scores; they measure how well a model recovers these thresholds,
not real-world forecasting skill.

HOW TO RUN
----------
    python build_real_dataset.py

OUTPUT : data/raw/aquaalert_raw.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR    = Path("data/raw/sources")
NCR_FILE   = SRC_DIR / "flood_prediction_ncr_2016_2020.csv"
AEGIS_FILE = SRC_DIR / "aegis_flood_points.csv"
JICA_FILE  = SRC_DIR / "jica_2014_major_floods.csv"
OUT_FILE   = Path("data/raw/aquaalert_raw.csv")

# Label thresholds (PROTOTYPE ASSUMPTIONS)
MODERATE_RAIN_MM   = 20.0
MODERATE_LEVEL_M   = 2.5

# AEGIS flood-height code at or above which a point counts as "deep" flooding.
# Commonly read as waist-high; the code book is not shipped with the dataset.
AEGIS_DEEP_CODE = 3

# Share of a city's AEGIS points that are "deep" → static hazard level.
HAZARD_MEDIUM_SHARE = 0.30
HAZARD_HIGH_SHARE   = 0.50

# Approximate city-hall / city-centre coordinates (lat, lon) for every NCR LGU
# plus the Rizal towns on the eastern edge.  AEGIS points are assigned to the
# nearest centre (a Voronoi approximation of city boundaries); only points
# nearest to the four study cities are kept.  PROTOTYPE ASSUMPTION – replace
# with official PSA/NAMRIA boundary polygons for a production system.
CITY_CENTRES = {
    "Manila":         (14.5958, 120.9772),
    "Quezon City":    (14.6760, 121.0437),
    "Marikina":       (14.6507, 121.1029),
    "Pasig":          (14.5764, 121.0851),
    "Caloocan":       (14.6497, 120.9672),
    "Caloocan North": (14.7566, 121.0450),
    "Makati":         (14.5547, 121.0244),
    "Taguig":         (14.5176, 121.0509),
    "Mandaluyong":    (14.5794, 121.0359),
    "San Juan":       (14.6019, 121.0355),
    "Pasay":          (14.5378, 121.0014),
    "Paranaque":      (14.4793, 121.0198),
    "Las Pinas":      (14.4497, 120.9830),
    "Muntinlupa":     (14.4081, 121.0415),
    "Valenzuela":     (14.7011, 120.9830),
    "Malabon":        (14.6625, 120.9567),
    "Navotas":        (14.6667, 120.9417),
    "Pateros":        (14.5446, 121.0687),
    "Cainta":         (14.5786, 121.1222),
    "San Mateo":      (14.6969, 121.1219),
}


def banner(title: str) -> None:
    w = 65
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def load_ncr_timeseries(filepath: Path) -> pd.DataFrame:
    """Load the daily NCR series and add per-city lag / rolling features."""
    banner("STEP 1 – Load NCR daily flood series")

    df = pd.read_csv(filepath, parse_dates=["Date"])
    print(f"  Rows: {len(df):,}   Cities: {sorted(df['Location'].unique())}")
    print(f"  Dates: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Recorded floods: {int(df['FloodOccurrence'].sum())}")

    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)
    g  = df.groupby("Location")

    # Only PAST and SAME-DAY values are used, so nothing leaks from the future.
    # The first day(s) of each city have no history → left as NaN on purpose;
    # preprocess.py fills them (median) and logs it in the quality report.
    df["rainfall_prev_day_mm"]    = g["Rainfall_mm"].shift(1)
    df["rainfall_last_3d_mm"]     = g["Rainfall_mm"].transform(
        lambda s: s.rolling(3, min_periods=3).sum())
    df["rainfall_last_7d_mm"]     = g["Rainfall_mm"].transform(
        lambda s: s.rolling(7, min_periods=7).sum())
    df["water_level_change_1d_m"] = g["WaterLevel_m"].diff()
    return df


def build_aegis_hazard(filepath: Path, cities: list[str]) -> pd.DataFrame:
    """Summarise AEGIS flood-height points into one hazard row per city."""
    banner("STEP 2 – Derive static hazard levels from AEGIS")

    pts = pd.read_csv(filepath)
    names   = np.array(list(CITY_CENTRES))
    centres = np.array(list(CITY_CENTRES.values()))

    # Equirectangular distance is plenty accurate at Metro Manila's scale.
    coslat = np.cos(np.radians(14.6))
    d2 = ((pts[["lat"]].values - centres[:, 0]) ** 2
          + ((pts[["lon"]].values - centres[:, 1]) * coslat) ** 2)
    pts["city"] = names[d2.argmin(axis=1)]

    pts = pts[pts["city"].isin(cities)]
    summary = pts.groupby("city").agg(
        aegis_points=("flood_heig", "size"),
        aegis_mean_flood_height_code=("flood_heig", "mean"),
        aegis_share_deep_flood=("flood_heig",
                                lambda h: (h >= AEGIS_DEEP_CODE).mean()),
        aegis_mean_elevation_m=("elevation", "mean"),
    ).round(4)

    summary["flood_hazard_level"] = np.select(
        [summary["aegis_share_deep_flood"] >= HAZARD_HIGH_SHARE,
         summary["aegis_share_deep_flood"] >= HAZARD_MEDIUM_SHARE],
        ["High", "Medium"],
        default="Low",
    )
    print(summary.to_string())
    return summary.reset_index().rename(columns={"city": "municipality"})


def assign_risk_label(df: pd.DataFrame) -> pd.Series:
    """Map the binary flood label onto Low / Moderate / High."""
    near_miss = ((df["Rainfall_mm"] >= MODERATE_RAIN_MM)
                 | (df["WaterLevel_m"] >= MODERATE_LEVEL_M))
    return pd.Series(
        np.select([df["FloodOccurrence"] == 1, near_miss],
                  ["High", "Moderate"], default="Low"),
        index=df.index,
    )


def jica_context_check(df: pd.DataFrame, filepath: Path) -> None:
    """Compare training-data rainfall against documented major NCR floods."""
    banner("STEP 4 – Context check against JICA (2014) major floods")

    jica = pd.read_csv(filepath)
    print(jica[["year", "storm", "sto_nino_peak_wl_m",
                "avg_watershed_rainfall_mm_1day"]].to_string(index=False))
    max_rain = df["rainfall_last_24h_mm"].max()
    min_major = jica["avg_watershed_rainfall_mm_1day"].min()
    print(f"\n  Max daily rainfall in training data : {max_rain:.1f} mm")
    print(f"  Smallest documented major flood     : {min_major:.1f} mm/day")
    if max_rain < min_major:
        print("  ⚠  No typhoon-scale rainfall day exists in the training data.\n"
              "     The models have never seen an Ondoy-type event.")


def main() -> None:
    print("\n" + "=" * 65)
    print("  AquaAlert – Build raw dataset from real sources")
    print("=" * 65)

    ncr    = load_ncr_timeseries(NCR_FILE)
    hazard = build_aegis_hazard(AEGIS_FILE, sorted(ncr["Location"].unique()))

    banner("STEP 3 – Merge and label")
    df = ncr.rename(columns={
        "Date":             "timestamp",
        "Location":         "municipality",
        "Rainfall_mm":      "rainfall_last_24h_mm",
        "WaterLevel_m":     "river_water_level_m",
        "SoilMoisture_pct": "soil_moisture_pct",
        "Elevation_m":      "elevation_m",
        "FloodOccurrence":  "flood_occurrence",
    })
    df["flood_risk"] = assign_risk_label(ncr)
    df = df.merge(hazard, on="municipality", how="left")

    df = df.sort_values(["timestamp", "municipality"]).reset_index(drop=True)
    df.insert(0, "observation_id", [f"OBS-{i + 1:05d}" for i in range(len(df))])

    df = df[[
        "observation_id", "timestamp", "municipality",
        "rainfall_last_24h_mm", "rainfall_prev_day_mm",
        "rainfall_last_3d_mm", "rainfall_last_7d_mm",
        "river_water_level_m", "water_level_change_1d_m",
        "soil_moisture_pct", "elevation_m",
        "aegis_points", "aegis_mean_flood_height_code",
        "aegis_share_deep_flood", "aegis_mean_elevation_m",
        "flood_hazard_level",
        "flood_occurrence", "flood_risk",
    ]]

    counts = df["flood_risk"].value_counts()
    for lbl in ["Low", "Moderate", "High"]:
        n = counts.get(lbl, 0)
        print(f"  {lbl:<10} {n:>5,} rows  ({100 * n / len(df):.1f}%)")

    jica_context_check(df, JICA_FILE)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n  Wrote {len(df):,} rows × {len(df.columns)} columns → {OUT_FILE}")


if __name__ == "__main__":
    main()
