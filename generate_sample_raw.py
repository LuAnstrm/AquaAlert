import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 300  # number of rows

barangays = [
    ("Sto. Niño",     "Marikina",   0.8),
    ("Tumana",        "Marikina",   0.7),
    ("Dela Paz",      "Pasig",      0.5),
    ("Ugong Norte",   "Pasig",      0.6),
    ("Bagumbayan",    "Quezon City", 0.3),
    ("Batasan Hills", "Quezon City", 0.4),
    ("Pandacan",      "Manila",     0.7),
    ("Paco",          "Manila",     0.6),
]

records = []
for i in range(N):
    bgy, muni, risk_weight = barangays[i % len(barangays)]
    r3h  = float(rng.exponential(scale=30) * risk_weight)
    r24h = float(r3h + rng.exponential(scale=60))
    fc   = float(rng.exponential(scale=20) * risk_weight)
    wl   = float(rng.uniform(8, 24)) if rng.random() > 0.2 else None
    wl_change = float(rng.normal(0.5, 1.0)) if wl is not None else None
    elev = float(rng.integers(3, 50))
    haz  = rng.choice(["Low", "Medium", "High"], p=[0.4, 0.35, 0.25])
    drain = rng.choice(["Clean", "Partly Blocked", "Blocked"], p=[0.45, 0.35, 0.20])
    count = int(rng.integers(0, 40))
    depth = float(rng.uniform(0, 80)) if count > 0 else 0.0

    # Simple proxy label
    score = r3h / 100 * 0.35 + (r24h / 200 * 0.30) + (count / 40 * 0.15)
    if wl is not None:
        score += (wl / 22) * 0.20
    if score > 0.65:
        label = "High"
    elif score > 0.33:
        label = "Moderate"
    else:
        label = "Low"

    records.append({
        "observation_id":  f"OBS-{i+1:04d}",
        "timestamp":       f"2024-{rng.integers(6,12):02d}-{rng.integers(1,28):02d} "
                           f"{rng.integers(0,23):02d}:{rng.integers(0,59):02d}:00",
        "barangay":        bgy,
        "municipality":    muni,
        "rainfall_last_3h_mm":          round(r3h, 1),
        "rainfall_last_24h_mm":         round(r24h, 1),
        "forecast_rainfall_next_3h_mm": round(fc, 1),
        "river_water_level_m":          round(wl, 2) if wl else None,
        "water_level_change_1h_m":      round(wl_change, 2) if wl_change else None,
        "elevation_m":                  elev,
        "flood_hazard_level":           haz,
        "drainage_condition":           drain,
        "citizen_report_count":         count,
        "average_reported_flood_depth_cm": round(depth, 1),
        "flood_risk":                   label,
    })

df = pd.DataFrame(records)

df = pd.concat([df, df.iloc[:5]], ignore_index=True)

for col in ["rainfall_last_3h_mm", "citizen_report_count"]:
    idx = rng.choice(df.index, 4, replace=False)
    df.loc[idx, col] = None


df.loc[0, "flood_hazard_level"]  = "Very High"
df.loc[1, "drainage_condition"]  = "Unknown"
df.loc[2, "flood_risk"]          = "Extreme"

df.to_csv("data/raw/aquaalert_raw.csv", index=False)
print(f"Wrote {len(df)} rows to data/raw/aquaalert_raw.csv")
print("(Includes 5 duplicates, 8 missing values, 3 invalid categories)")
