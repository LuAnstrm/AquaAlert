"""Measure non-rain flood drivers and test whether tide data improves the model."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import tide
from common import DAILY_FILE, HOLDOUT_YEAR, REPORT_DIR, build_features, build_labels, load_catalog
from improve import (BASELINES, FEATURE_SETS, bootstrap_gain, development_gates, logistic)
from train import evaluate, leave_one_year_out, model_factories

# Screening limits for labelling events. They describe the record and are not
# flood thresholds. Percentiles use wet-season days (June-November).
LOW_LOCAL_RAIN_3D_MM = 50.0
HIGH_PERCENTILE = 0.90
LOOKBACK_DAYS = 3


def check_tide_model() -> dict:
    model = tide.fit()
    observed = tide.load_observed()
    test = observed.loc["2026-01-01":]
    predicted = tide.predict_hourly(test.index, model)
    residual = test - predicted
    result = {"test_hours": len(test), "rmse_m": round(float(np.sqrt((residual ** 2).mean())), 3),
              "correlation": round(float(np.corrcoef(test, predicted)[0, 1]), 3)}
    print(f"Tide model, unseen 2026: {result}")
    return result


def event_drivers(catalog: pd.DataFrame, daily: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Compare conditions before floods with ordinary wet-season windows."""
    observed = tide.load_observed()
    # Remove the long-term sea-level trend so recent years are not always "high".
    observed = observed - (tide.predict_hourly(observed.index) - tide.predict_periodic(observed.index))
    daily_tide = observed.groupby((observed.index + pd.Timedelta(hours=8)).normalize()).max()

    dates = features.index
    frame = pd.DataFrame({
        "local_rain": features["rain_3d_local_mean"],
        "upstream_rain": features["rain_7d_upstream_mean"],
        "river_flow": daily["discharge_pampanga_main"].reindex(dates),
        "high_tide": daily_tide.reindex(dates),
    })
    rolled = frame.rolling(LOOKBACK_DAYS + 1, min_periods=1).max()
    wet = dates.month.isin([6, 7, 8, 9, 10, 11])
    limits = frame[wet].quantile(HIGH_PERCENTILE)

    events = catalog[catalog["kind"] == "event"]
    table = rolled.reindex(events["start"]).round(2).reset_index(names="start")
    table.insert(1, "name", events["name"].to_numpy())
    table.insert(2, "date_basis", events["date_basis"].to_numpy())
    table["low_local_rain"] = table["local_rain"] < LOW_LOCAL_RAIN_3D_MM
    for column in ["upstream_rain", "river_flow", "high_tide"]:
        table[f"is_high_{column}"] = table[column] >= limits[column]
    table.to_csv(REPORT_DIR / "event_drivers.csv", index=False)

    print("Shares of flood starts vs ordinary wet-season windows")
    for column in ["upstream_rain", "river_flow", "high_tide"]:
        scored = table[column].notna()
        share = float(table.loc[scored, f"is_high_{column}"].mean())
        ordinary = rolled.loc[wet, column].dropna()
        base = float((ordinary >= limits[column]).mean())
        print(f"  {column}: floods {share:.0%} (n={int(scored.sum())}), ordinary windows {base:.0%}")
    low = table[table["low_local_rain"]]
    print(f"Floods with under {LOW_LOCAL_RAIN_3D_MM:.0f} mm modelled local rain in 3 days: "
          f"{len(low)} of {len(table)}")
    print(low[["start", "name", "local_rain", "upstream_rain", "river_flow", "high_tide"]].to_string(index=False))
    return table


def tide_test(catalog, features, labels) -> pd.DataFrame:
    """Run the one pre-declared comparison, using development years only."""
    usable = labels["usable"] & (features.index.year < HOLDOUT_YEAR)
    X, y = features[usable], labels.loc[usable, "target"].astype(int)
    rows, baseline_scores = [], {}
    for name in BASELINES:
        score, _ = leave_one_year_out(model_factories()[name], X, y)
        baseline_scores[name] = score.loc[y.index]
    best = max(baseline_scores, key=lambda name: average_precision_score(y, baseline_scores[name]))

    sets = {"compact (no tide)": FEATURE_SETS["compact"],
            "compact + predicted tide": FEATURE_SETS["compact"] + ["tide_high_next2d"]}
    for name, columns in sets.items():
        score, alerts = leave_one_year_out(logistic(0.05), X[columns], y)
        score = score.loc[y.index]
        row, _ = evaluate(name, y, score, alerts, catalog, labels, "dev")
        low, high = bootstrap_gain(y, score, baseline_scores[best])
        gates = development_gates(row, low)
        row.update(gain_ci_low=low, gain_ci_high=high, gates_passed=sum(gates.values()), **gates)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "tide_comparison.csv", index=False)
    print(table[["model", "pr_auc", "events_detected", "events", "documented_warned_before_start",
                 "documented_events", "false_alert_episodes_per_year", "alert_days_per_year",
                 "gain_ci_low", "gain_ci_high", "gates_passed"]].T.to_string())
    return table


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    check_tide_model()
    catalog = load_catalog()
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
    features = build_features(daily).dropna()
    labels = build_labels(features.index, catalog)
    event_drivers(catalog, daily, features)
    tide_test(catalog, features, labels)


if __name__ == "__main__":
    main()
