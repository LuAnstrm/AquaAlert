"""Test each flood factor group, then compare Hagonoy and Calumpit."""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import tide
from common import (DAILY_FILE, HAZARD_FILE, HOLDOUT_YEAR, RAIN_POINTS, REPORT_DIR, ROOT, build_features,
                    build_labels, load_catalog)
from improve import FEATURE_SETS, bootstrap_gain, development_gates, logistic
from train import evaluate, fit_with_threshold, leave_one_year_out, model_factories, score_events

PROCESSED = ROOT / "data/processed"
CANDIDATE_FILE = ROOT / "models/bulacan_all_factors_candidate.pkl"
FORECAST_CANDIDATE_FILE = ROOT / "models/bulacan_rain_forecast_candidate.pkl"
LOCAL = ["calumpit", "malolos", "sjdm"]
UPSTREAM = ["san_miguel", "angat_watershed", "upper_pampanga"]
TC_FAR_KM = 2000.0
FORECAST_YEARS = [2024, 2025, 2026]
TOWNS = ["Hagonoy", "Calumpit"]
HIGH_PERCENTILE = 0.90
LOOKBACK_DAYS = 3
WET_MONTHS = [6, 7, 8, 9, 10, 11]

BASE = FEATURE_SETS["compact"]
FAMILIES = {
    "rain_forecast": ["fc_rain_local_next2d", "fc_rain_upstream_next2d", "fc_rain_max_next2d"],
    "tide": ["tide_high_next2d", "spring_tide_index"],
    "wind_pressure": ["onshore_wind_max_3d", "pressure_min_3d"],
    "cyclone": ["tc_distance_2d"],
    "el_nino": ["oni_lag2"],
    "rain_intensity_soil": ["rain_peak_hour_3d", "rain_hours_3d", "soil_deep"],
}
NO_FORECAST = [c for name, cols in FAMILIES.items() if name != "rain_forecast" for c in cols]
LIVE_MISSING = {"soil_deep"}


def add_factors(base, weather, cyclones, oni, forecast) -> pd.DataFrame:
    out = base.copy()
    index = out.index
    weather = weather.reindex(index)
    out["onshore_wind_max_3d"] = weather["onshore_wind_max_kmh"].rolling(3, min_periods=3).max()
    out["pressure_min_3d"] = weather["pressure_min_hpa"].rolling(3, min_periods=3).min()
    out["rain_peak_hour_3d"] = weather["rain_peak_hour_mm"].rolling(3, min_periods=3).max()
    out["rain_hours_3d"] = weather["rain_hours"].rolling(3, min_periods=3).sum()
    if "soil_deep" in weather:
        out["soil_deep"] = weather["soil_deep"]
    distance = cyclones["tc_distance_km"].reindex(index).fillna(TC_FAR_KM).clip(upper=TC_FAR_KM)
    out["tc_distance_2d"] = distance.rolling(2, min_periods=1).min()
    # ONI arrives after the month ends, so use the value from two months earlier.
    lagged_months = (index.to_period("M") - 2).to_timestamp()
    out["oni_lag2"] = oni["oni"].reindex(lagged_months).to_numpy()
    for name, values in forecast.items():
        out[name] = values.reindex(index)
    return out


def proxy_forecast(daily) -> dict:
    """Use later observed rain as a best-case stand-in for a perfect forecast."""
    local = daily[[f"rain_{p}" for p in LOCAL]].mean(axis=1)
    upstream = daily[[f"rain_{p}" for p in UPSTREAM]].mean(axis=1)
    peak = daily[[f"rain_{p}" for p in RAIN_POINTS]].max(axis=1)
    return {"fc_rain_local_next2d": local.shift(-1) + local.shift(-2),
            "fc_rain_upstream_next2d": upstream.shift(-1) + upstream.shift(-2),
            "fc_rain_max_next2d": pd.concat([peak.shift(-1), peak.shift(-2)], axis=1).max(axis=1)}


def real_forecast(forecasts) -> dict:
    """Use only forecasts issued 1 day before day t+1 and 2 days before day t+2."""
    day1 = forecasts.filter(like="fc1_").rename(columns=lambda c: c[4:]).shift(-1)
    day2 = forecasts.filter(like="fc2_").rename(columns=lambda c: c[4:]).shift(-2)
    both = day1.add(day2)
    return {"fc_rain_local_next2d": both[LOCAL].mean(axis=1),
            "fc_rain_upstream_next2d": both[UPSTREAM].mean(axis=1),
            "fc_rain_max_next2d": pd.concat([day1.max(axis=1), day2.max(axis=1)], axis=1).max(axis=1)}


def town_catalog(catalog, town) -> pd.DataFrame:
    """Keep floods naming this town, and mask days when flood reports do not name it."""
    out = catalog[catalog["kind"].isin(["event", "mask", "tidal_event"])].copy()
    named = out["towns"].fillna("").str.split(";").apply(lambda towns: town in towns)
    flood_rows = out["kind"].isin(["event", "tidal_event"])
    out.loc[flood_rows & named, "kind"] = "event"
    out.loc[flood_rows & ~named & (out["kind"] == "event"), "kind"] = "mask"
    return out[out["kind"].isin(["event", "mask"])]


def run_family_tests(features, catalog) -> tuple[pd.DataFrame, dict]:
    labels = build_labels(features.index, catalog)
    usable = labels["usable"] & (features.index.year < HOLDOUT_YEAR)
    X, y = features[usable], labels.loc[usable, "target"].astype(int)
    print(f"Development: {len(X):,} days, {int(y.sum())} target days")

    baseline_scores = {}
    for name in ["rule_river_discharge", "rule_3day_rain"]:
        score, alerts = leave_one_year_out(model_factories()[name], X, y)
        baseline_scores[name] = score.loc[y.index]
    best_rule = max(baseline_scores, key=lambda n: average_precision_score(y, baseline_scores[n]))

    sets = {"base": BASE}
    sets.update({f"base+{name}": BASE + cols for name, cols in FAMILIES.items()})
    sets["all_available_at_issue"] = BASE + NO_FORECAST
    sets["all_with_perfect_rain"] = BASE + NO_FORECAST + FAMILIES["rain_forecast"]

    rows = []
    for name, columns in sets.items():
        score, alerts = leave_one_year_out(logistic(0.05), X[columns], y)
        row, _ = evaluate(name, y, score.loc[y.index], alerts, catalog, labels, "dev_2011_2025")
        low, high = bootstrap_gain(y, score.loc[y.index], baseline_scores[best_rule])
        gates = development_gates(row, low)
        row.update(gain_ci_low=low, gain_ci_high=high, gates_passed=sum(gates.values()), **gates)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["upper_bound_only"] = table["model"].isin(["base+rain_forecast", "all_with_perfect_rain"])
    return table, sets


def forecast_test(era5_features, live_base, live_weather, cyclones, oni, forecasts, catalog) -> pd.DataFrame:
    """Train on other years, then score each test year with real issued forecasts."""
    labels = build_labels(era5_features.index, catalog)
    usable = labels["usable"]
    X_all, y_all = era5_features[usable], labels.loc[usable, "target"].astype(int)

    live = add_factors(live_base, live_weather, cyclones, oni, real_forecast(forecasts))
    live = live[live.index.year.isin(FORECAST_YEARS)]
    available = [c for c in NO_FORECAST if c not in LIVE_MISSING]
    sets = {"base": BASE, "base+rain_forecast": BASE + FAMILIES["rain_forecast"],
            "all_factors_live": BASE + available + FAMILIES["rain_forecast"]}

    rows, event_tables = [], []
    for name, columns in sets.items():
        scored = live[columns].dropna()
        live_labels = build_labels(scored.index, catalog)
        scored = scored[live_labels["usable"]]
        y = live_labels.loc[scored.index, "target"].astype(int)
        scores, flags = [], []
        for year in FORECAST_YEARS:
            train = X_all.index.year != year
            model, threshold = fit_with_threshold(logistic(0.05), X_all.loc[train, columns], y_all[train])
            test = scored[scored.index.year == year]
            result = pd.Series(model.predict_proba(test)[:, 1], index=test.index)
            scores.append(result)
            flags.append(result >= threshold)
        row, events = evaluate(name, y, pd.concat(scores), pd.concat(flags), catalog, live_labels,
                               "real_forecast_2024_2026")
        rows.append(row)
        event_tables.append(events.assign(model=name))
    pd.concat(event_tables).to_csv(REPORT_DIR / "round2_forecast_events.csv", index=False)
    return pd.DataFrame(rows)


def forecast_skill(daily, forecasts) -> pd.DataFrame:
    rows = []
    for point in RAIN_POINTS:
        observed = daily[f"rain_{point}"]
        for lead in [1, 2]:
            forecast = forecasts[f"fc{lead}_{point}"]
            pair = pd.concat([observed, forecast], axis=1, keys=["obs", "fc"]).dropna()
            heavy = pair["obs"] >= 50
            rows.append({"point": point, "lead_days": lead, "days": len(pair),
                         "correlation": round(float(pair["obs"].corr(pair["fc"])), 2),
                         "mean_obs_on_heavy_days": round(float(pair.loc[heavy, "obs"].mean()), 1),
                         "mean_fc_on_heavy_days": round(float(pair.loc[heavy, "fc"].mean()), 1),
                         "heavy_days": int(heavy.sum()),
                         "heavy_days_with_fc_25mm": int((pair.loc[heavy, "fc"] >= 25).sum())})
    return pd.DataFrame(rows)


def town_models(features, catalog) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, event_tables = [], []
    for town in TOWNS:
        town_events = town_catalog(catalog, town)
        labels = build_labels(features.index, town_events)
        usable = labels["usable"] & (features.index.year < HOLDOUT_YEAR)
        X, y = features[usable], labels.loc[usable, "target"].astype(int)
        for name, columns in {"base": BASE, "all_available_at_issue": BASE + NO_FORECAST}.items():
            score, alerts = leave_one_year_out(logistic(0.05), X[columns], y)
            row, events = evaluate(f"{town}: {name}", y, score.loc[y.index], alerts, town_events,
                                   labels, "town_dev_2011_2025")
            rows.append(row)
            event_tables.append(events.assign(town=town, model=name))
        rule, rule_alerts = leave_one_year_out(model_factories()["rule_river_discharge"], X, y)
        row, _ = evaluate(f"{town}: river rule", y, rule.loc[y.index], rule_alerts, town_events,
                          labels, "town_dev_2011_2025")
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(event_tables)


def town_drivers(features, daily, catalog) -> pd.DataFrame:
    observed = tide.load_observed()
    observed = observed - (tide.predict_hourly(observed.index) - tide.predict_periodic(observed.index))
    daily_tide = observed.groupby((observed.index + pd.Timedelta(hours=8)).normalize()).max()
    frame = pd.DataFrame({
        "local_rain": features["rain_3d_local_mean"],
        "upstream_rain": features["rain_7d_upstream_mean"],
        "river_flow": daily["discharge_pampanga_main"].reindex(features.index),
        "angat_flow": daily["discharge_angat_lower"].reindex(features.index),
        "high_tide": daily_tide.reindex(features.index),
    })
    wet = frame.index.month.isin(WET_MONTHS)
    limits = frame[wet].quantile(HIGH_PERCENTILE)
    rolled = frame.rolling(LOOKBACK_DAYS + 1, min_periods=1).max()
    ordinary = {c: float((rolled.loc[wet, c].dropna() >= limits[c]).mean()) for c in frame}
    rows = []
    for town in TOWNS:
        named = catalog["towns"].fillna("").str.split(";").apply(lambda towns: town in towns)
        for kinds, group in [(["event"], "storm_and_other_reports"), (["tidal_event"], "tide_reports")]:
            starts = catalog.loc[named & catalog["kind"].isin(kinds), "start"]
            picks = rolled.reindex(starts).dropna(how="all")
            if picks.empty:
                continue
            row = {"town": town, "group": group, "reports": len(picks)}
            for column in frame:
                row[f"{column}_share"] = round(float((picks[column].dropna() >= limits[column]).mean()), 2)
            row["low_local_rain_share"] = round(float((picks["local_rain"] < 50).mean()), 2)
            rows.append(row)
    rows.append({"town": "ordinary wet-season windows", "group": "comparison", "reports": None,
                 **{f"{c}_share": round(v, 2) for c, v in ordinary.items()}})
    return pd.DataFrame(rows)


def tide_rule_check(catalog) -> pd.DataFrame:
    """Measure how often Manila predicted tide reaches levels seen in Hagonoy tide reports."""
    reports = catalog[catalog["kind"] == "tidal_event"]
    dates = pd.date_range("2023-01-01", "2026-12-31")
    times = pd.date_range(dates.min() - pd.Timedelta(hours=8), dates.max() + pd.Timedelta(hours=16), freq="h")
    predicted = tide.predict_hourly(times)
    daily = predicted.groupby((predicted.index + pd.Timedelta(hours=8)).normalize()).max()
    rows = []
    for row in reports.itertuples():
        level = float(daily.get(row.start, np.nan))
        rows.append({"date": row.start.date(), "name": row.name, "predicted_high_m": round(level, 2),
                     "days_per_year_at_or_above": round(float((daily >= level).sum()) / 4, 0)})
    return pd.DataFrame(rows)


def site_comparison(catalog, town_results) -> pd.DataFrame:
    hazard = pd.read_csv(HAZARD_FILE)
    static_file = PROCESSED / "bulacan_barangay_static.csv"
    static = pd.read_csv(static_file).drop(columns=["adm4_name", "adm3_name"]) if static_file.exists() else None
    table = hazard.merge(static, on="adm4_pcode", how="left") if static is not None else hazard
    rows = []
    for town in TOWNS:
        part = table[table["adm3_name"] == town]
        named = catalog["towns"].fillna("").str.split(";").apply(lambda towns: town in towns)
        row = {"town": town, "barangays": len(part),
               "area_km2": round(float(part["area_m2"].sum() / 1e6), 1),
               "flood_reports": int((named & (catalog["kind"] == "event")).sum()),
               "tide_reports": int((named & (catalog["kind"] == "tidal_event")).sum()),
               "noah_medium_high_25yr": round(float(((part["share_medium_25yr"] + part["share_high_25yr"]) * part["area_m2"]).sum()
                                                    / part["area_m2"].sum()), 3)}
        for column in ["elev_mean_m", "elev_p10_m", "share_below_2m", "share_water", "share_built",
                       "share_wetland_mangrove", "waterway_km_per_km2"]:
            if column in part:
                row[column] = round(float((part[column] * part["area_m2"]).sum() / part["area_m2"].sum()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
    weather = pd.read_csv(PROCESSED / "bulacan_weather_factors.csv", parse_dates=["date"], index_col="date")
    live_weather = pd.read_csv(PROCESSED / "bulacan_weather_factors_live.csv", parse_dates=["date"], index_col="date")
    cyclones = pd.read_csv(PROCESSED / "bulacan_cyclones.csv", parse_dates=["date"], index_col="date")
    oni = pd.read_csv(PROCESSED / "enso_oni.csv", parse_dates=["month"], index_col="month")
    forecasts = pd.read_csv(PROCESSED / "bulacan_rain_forecasts.csv", parse_dates=["date"], index_col="date")
    live_daily = pd.read_csv(DAILY_FILE.with_name("bulacan_daily_forecast_archive.csv"), parse_dates=["date"], index_col="date")

    print("Cyclone tracks through", cyclones["tracks_available_until"].iloc[0])
    print("Forecast coverage", forecasts.dropna().index.min().date(), forecasts.dropna().index.max().date())
    base = build_features(daily).dropna()
    features = add_factors(base, weather, cyclones, oni, proxy_forecast(daily)).dropna()

    table, sets = run_family_tests(features, catalog)
    table.to_csv(REPORT_DIR / "round2_factor_tests.csv", index=False)
    print(table[["model", "pr_auc", "events_detected", "events", "documented_warned_before_start",
                 "documented_events", "false_alert_episodes_per_year", "alert_days_per_year",
                 "gain_ci_low", "gain_ci_high", "gates_passed", "upper_bound_only"]].to_string(index=False))

    skill = forecast_skill(daily, forecasts)
    skill.to_csv(REPORT_DIR / "round2_forecast_skill.csv", index=False)
    print("\nForecast skill against ERA5:\n" + skill.to_string(index=False))

    live_base = build_features(live_daily).dropna()
    real = forecast_test(features, live_base, live_weather, cyclones, oni, forecasts, catalog)
    real.to_csv(REPORT_DIR / "round2_real_forecast_test.csv", index=False)
    print("\nReal forecasts, 2024-2026:\n" + real[["model", "days", "flood_target_days", "pr_auc", "events_detected",
                                                    "events", "documented_warned_before_start", "documented_events",
                                                    "false_alert_episodes_per_year", "alert_days_per_year"]].to_string(index=False))
    events = pd.read_csv(REPORT_DIR / "round2_forecast_events.csv")
    print(events[["model", "start", "name", "alert_before_start", "alert_by_end"]].to_string(index=False))

    towns, town_events = town_models(features, catalog)
    towns.to_csv(REPORT_DIR / "round2_town_models.csv", index=False)
    town_events.to_csv(REPORT_DIR / "round2_town_events.csv", index=False)
    print("\nTown models:\n" + towns[["model", "days", "flood_target_days", "pr_auc", "roc_auc", "events_detected",
                                       "events", "documented_warned_before_start", "documented_events",
                                       "false_alert_episodes_per_year", "alert_days_per_year"]].to_string(index=False))

    drivers = town_drivers(features, daily, catalog)
    drivers.to_csv(REPORT_DIR / "round2_town_drivers.csv", index=False)
    print("\nTown flood drivers:\n" + drivers.to_string(index=False))

    tide_rule = tide_rule_check(catalog)
    tide_rule.to_csv(REPORT_DIR / "round2_hagonoy_tide_levels.csv", index=False)
    print("\nHagonoy tide reports:\n" + tide_rule.to_string(index=False))

    sites = site_comparison(catalog, towns)
    sites.to_csv(REPORT_DIR / "site_comparison.csv", index=False)
    print("\nSites:\n" + sites.T.to_string())

    # Keep this file separate because live serving is not built and it failed the gates unless shown otherwise.
    available = table[(~table["upper_bound_only"])]
    labels = build_labels(features.index, catalog)
    usable = labels["usable"]
    columns = BASE + NO_FORECAST
    model, threshold = fit_with_threshold(logistic(0.05), features.loc[usable, columns],
                                          labels.loc[usable, "target"].astype(int))
    joblib.dump({"model": model, "threshold": threshold, "name": "all_available_at_issue", "features": columns,
                 "status": "candidate only; not used by predict.py"}, CANDIDATE_FILE)
    forecast_columns = BASE + FAMILIES["rain_forecast"]
    forecast_model, forecast_threshold = fit_with_threshold(logistic(0.05), features.loc[usable, forecast_columns],
                                                            labels.loc[usable, "target"].astype(int))
    joblib.dump({"model": forecast_model, "threshold": forecast_threshold, "name": "base+rain_forecast",
                 "features": forecast_columns, "status": "shadow only; failed acceptance gates"},
                FORECAST_CANDIDATE_FILE)
    print(f"\nPassed all five gates: {sorted(available.loc[available['gates_passed'] == 5, 'model'])}")
    print(f"Saved candidate to {CANDIDATE_FILE}")


if __name__ == "__main__":
    main()
