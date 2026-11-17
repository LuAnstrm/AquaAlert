"""Run the limited improvement round defined in ACCEPTANCE_CRITERIA.md."""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (DAILY_FILE, HOLDOUT_YEAR, MODEL_FILE, REPORT_DIR, build_features,
                    build_labels, load_catalog)
from fetch_history import ARCHIVE_URL, collect
from train import (SEED, evaluate, fit_with_threshold, leave_one_year_out, model_factories,
                   score_events)

LIVE_FILE = DAILY_FILE.with_name("bulacan_daily_forecast_archive.csv")
BOOTSTRAPS = 2000
BASELINES = ["rule_3day_rain", "rule_river_discharge", "monthly_climatology"]

SEASON = ["season_sin", "season_cos"]
RAIN_SUMMARY = ["rain_1d_max_all", "rain_3d_local_mean", "rain_7d_local_mean",
                "rain_3d_upstream_mean", "rain_7d_upstream_mean", "rain_14d_upstream_mean"]
RIVER_MAIN = ["discharge_pampanga_main", "discharge_rise_3d_pampanga_main",
              "discharge_peak_7d_pampanga_main", "discharge_angat_lower",
              "discharge_rise_3d_angat_lower", "discharge_peak_7d_angat_lower"]
FEATURE_SETS = {
    "compact": RAIN_SUMMARY + RIVER_MAIN + ["soil_moisture_calumpit"] + SEASON,
    "compact_tide": RAIN_SUMMARY + RIVER_MAIN + ["soil_moisture_calumpit", "spring_tide_index"] + SEASON,
    "river_only": RIVER_MAIN + SEASON,
}


class Blend:
    """Average the rank-free probabilities of a linear model and a tree model."""

    def __init__(self):
        self.linear = logistic(0.05)()
        self.trees = forest()

    def fit(self, X, y):
        self.linear.fit(X, y)
        self.trees.fit(X, y)
        return self

    def predict_proba(self, X):
        return (self.linear.predict_proba(X) + self.trees.predict_proba(X)) / 2


def logistic(c):
    return lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED),
    )


def forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                  class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)


def gbm():
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=5,
                              min_child_samples=15, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=10.0, scale_pos_weight=10.0,
                              random_state=SEED, verbose=-1, n_jobs=1)


def variants() -> dict:
    """Return 12 combinations, within the limit of 15 set in the criteria."""
    models = {"lr_c005": logistic(0.05), "lr_c02": logistic(0.2), "forest": forest, "blend": Blend}
    return {f"{name}|{features}": (factory, FEATURE_SETS[features])
            for features in FEATURE_SETS for name, factory in models.items()}


def bootstrap_gain(y, score, baseline):
    """Compare PR-AUC by resampling whole years, because days within a year correlate."""
    rng = np.random.default_rng(SEED)
    years = np.unique(y.index.year)
    groups = {year: np.flatnonzero(y.index.year == year) for year in years}
    gains = []
    for _ in range(BOOTSTRAPS):
        sample = np.concatenate([groups[year] for year in rng.choice(years, len(years))])
        if y.iloc[sample].sum() == 0:
            continue
        gains.append(average_precision_score(y.iloc[sample], score.iloc[sample])
                     - average_precision_score(y.iloc[sample], baseline.iloc[sample]))
    return tuple(np.percentile(gains, [2.5, 97.5]).round(3))


def development_gates(row, low_gain) -> dict:
    return {
        "D1_detection": row["events_detected"] / row["events"] >= 0.80,
        "D2_early": row["documented_warned_before_start"] / row["documented_events"] >= 0.75,
        "D3_false_alarms": row["false_alert_episodes_per_year"] <= 3,
        "D4_alert_burden": row["alert_days_per_year"] <= 25,
        "D5_beats_rules": low_gain > 0,
    }


def main() -> None:
    catalog = load_catalog()
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
    features = build_features(daily).dropna()
    labels = build_labels(features.index, catalog)
    usable = labels["usable"]
    X_all, y_all = features[usable], labels.loc[usable, "target"].astype(int)
    dev = X_all.index.year < HOLDOUT_YEAR
    X_dev, y_dev = X_all[dev], y_all[dev]
    print(f"Development: {len(X_dev):,} days, {int(y_dev.sum())} target days")

    rows, baseline_scores = [], {}
    for name in BASELINES:
        score, alerts = leave_one_year_out(model_factories()[name], X_dev, y_dev)
        row, _ = evaluate(name, y_dev, score.loc[y_dev.index], alerts, catalog, labels, "dev")
        rows.append(row)
        baseline_scores[name] = score.loc[y_dev.index]
    best_baseline = max(BASELINES, key=lambda n: next(r["pr_auc"] for r in rows if r["model"] == n))

    for name, (factory, columns) in variants().items():
        score, alerts = leave_one_year_out(factory, X_dev[columns], y_dev)
        score = score.loc[y_dev.index]
        row, _ = evaluate(name, y_dev, score, alerts, catalog, labels, "dev")
        low, high = bootstrap_gain(y_dev, score, baseline_scores[best_baseline])
        gates = development_gates(row, low)
        row.update(gain_ci_low=low, gain_ci_high=high, gates_passed=sum(gates.values()), **gates)
        rows.append(row)
        print(f"{name:28s} PR-AUC {row['pr_auc']:.3f} detect {row['events_detected']}/{row['events']} "
              f"early {row['documented_warned_before_start']}/{row['documented_events']} "
              f"false/yr {row['false_alert_episodes_per_year']} days/yr {row['alert_days_per_year']} "
              f"gain [{low},{high}] gates {row['gates_passed']}/5")

    results = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "improvement_round.csv", index=False)
    print("Best baseline:", best_baseline)
    print(results[results["model"].isin(BASELINES)][
        ["model", "pr_auc", "events_detected", "events", "documented_warned_before_start",
         "documented_events", "false_alert_episodes_per_year", "alert_days_per_year"]].to_string(index=False))

    candidates = results[results["gates_passed"].notna()]
    finalist = candidates.sort_values(["gates_passed", "pr_auc"], ascending=False).iloc[0]["model"]
    factory, columns = variants()[finalist]
    print(f"\nFinalist chosen from development years only: {finalist}")

    # T1: for each year, train without it and score forecast-archive inputs.
    live = build_features(pd.read_csv(LIVE_FILE, parse_dates=["date"], index_col="date")).dropna()
    live_labels = build_labels(live.index, catalog)
    live_X = live[live_labels["usable"] & live.index.year.isin([2023, 2024, 2025, 2026])]
    scores, alerts = [], []
    for year in sorted(set(live_X.index.year)):
        train = X_all.index.year != year
        model, threshold = fit_with_threshold(factory, X_all.loc[train, columns], y_all[train])
        test = live_X.loc[live_X.index.year == year, columns]
        score = pd.Series(model.predict_proba(test)[:, 1], index=test.index)
        scores.append(score)
        alerts.append(score >= threshold)
    live_y = live_labels.loc[live_X.index, "target"].astype(int)
    t1, t1_events = evaluate(finalist, live_y, pd.concat(scores), pd.concat(alerts), catalog,
                             live_labels, "T1_live_like")
    t1_detect = t1["events_detected"] / t1["events"]
    t1_pass = t1_detect >= 0.75 and t1["false_alert_episodes_per_year"] <= 4
    print(f"T1 live-like: {t1['events_detected']}/{t1['events']} detected, "
          f"{t1['false_alert_episodes_per_year']} false episodes/yr, PR-AUC {t1['pr_auc']}: "
          f"{'PASS' if t1_pass else 'FAIL'}")
    print(t1_events[["start", "name", "alert_before_start", "alert_by_end"]].to_string(index=False))

    # T2: 2009 is outside all training data and was never used for selection.
    model, threshold = fit_with_threshold(factory, X_all[columns], y_all)
    old = collect(ARCHIVE_URL, "2009-08-01", "2009-10-20", "era5")
    old_features = build_features(old).dropna()[columns]
    old_score = pd.Series(model.predict_proba(old_features)[:, 1], index=old_features.index)
    old_alerts = old_score >= threshold
    old_catalog = catalog[catalog["kind"] == "test_event"].assign(kind="event")
    t2_events = score_events(old_alerts, old_catalog)
    t2_pass = bool(t2_events["alert_by_end"].all())
    print(f"T2 unseen 2009: {'PASS' if t2_pass else 'FAIL'}")
    print(t2_events[["start", "name", "alert_before_start", "alert_by_end"]].to_string(index=False))
    print(f"2009 alert days in Aug-Oct window: {int(old_alerts.sum())}; "
          f"Ondoy-season top scores:\n{old_score.nlargest(6).round(3).to_string()}")

    verdict = pd.DataFrame([{"finalist": finalist, "T1_pass": t1_pass, "T2_pass": t2_pass,
                             "dev_gates_passed": int(results.set_index('model').loc[finalist, 'gates_passed']),
                             "variants_tried": len(variants())}])
    verdict.to_csv(REPORT_DIR / "acceptance_result.csv", index=False)
    print(verdict.T.to_string())

    joblib.dump({"model": model, "threshold": threshold, "name": finalist, "features": columns},
                MODEL_FILE)
    print(f"Saved {finalist} to {MODEL_FILE}")


if __name__ == "__main__":
    main()
