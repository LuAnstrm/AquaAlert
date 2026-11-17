"""Train and evaluate the Bulacan flood model against dated flood reports."""

from math import sqrt

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (DAILY_FILE, HOLDOUT_YEAR, REPORT_DIR, build_features, build_labels,
                    load_catalog)

SEED = 42
INNER_FOLDS = 4
EARLY_WINDOW_DAYS = 3
WET_MONTHS = [6, 7, 8, 9, 10, 11]
WET_SEASON_DAYS = 183
LIVE_ARCHIVE_FILE = DAILY_FILE.with_name("bulacan_daily_forecast_archive.csv")
LIVE_ARCHIVE_YEARS = [2023, 2024, 2025, 2026]


class ColumnScore:
    """Use one feature as a score, as a simple threshold rule would."""

    def __init__(self, column):
        self.column = column

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        score = X[self.column].to_numpy()
        return np.column_stack([1 - score, score])


class MonthlyRate:
    """Score each day by the historical flood rate for its calendar month."""

    def fit(self, X, y):
        rates = pd.Series(np.asarray(y, dtype=float)).groupby(X.index.month.to_numpy()).mean()
        self.rates = rates
        self.default = float(np.mean(y))
        return self

    def predict_proba(self, X):
        score = pd.Series(X.index.month).map(self.rates).fillna(self.default).to_numpy()
        return np.column_stack([1 - score, score])


def model_factories() -> dict:
    return {
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            scale_pos_weight=10.0, random_state=SEED, verbose=-1, n_jobs=1,
        ),
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000, random_state=SEED),
        ),
        "rule_3day_rain": lambda: ColumnScore("rain_3d_calumpit"),
        "rule_river_discharge": lambda: ColumnScore("discharge_pampanga_main"),
        "monthly_climatology": MonthlyRate,
    }


def best_f2_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Pick the threshold that favours catching floods over avoiding false alarms."""
    candidates = np.unique(np.quantile(score, np.linspace(0.5, 0.999, 300)))
    best, best_f2 = float(candidates[-1]), -1.0
    for threshold in candidates:
        predicted = score >= threshold
        true_positive = np.sum(predicted & y)
        precision = true_positive / max(predicted.sum(), 1)
        recall = true_positive / max(y.sum(), 1)
        f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
        if f2 > best_f2:
            best, best_f2 = float(threshold), f2
    return best


def fit_with_threshold(factory, X: pd.DataFrame, y: pd.Series):
    """Fit a model and choose its alert threshold from out-of-fold scores only."""
    years = X.index.year.to_numpy()
    folds = min(INNER_FOLDS, len(np.unique(years)))
    oof = np.zeros(len(X))
    for train_idx, valid_idx in GroupKFold(n_splits=folds).split(X, y, groups=years):
        model = factory().fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    threshold = best_f2_threshold(y.to_numpy(), oof)
    return factory().fit(X, y), threshold


def wilson_interval(hits: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def score_events(alerts: pd.Series, catalog: pd.DataFrame) -> pd.DataFrame:
    """Check each reported flood for an alert before it and during it."""
    rows = []
    events = catalog[(catalog["kind"] == "event")
                     & catalog["start"].dt.year.isin(alerts.index.year.unique())]
    for event in events.itertuples():
        before = alerts.loc[event.start - pd.Timedelta(days=EARLY_WINDOW_DAYS):
                            event.start - pd.Timedelta(days=1)]
        through_event = alerts.loc[event.start - pd.Timedelta(days=EARLY_WINDOW_DAYS):event.end]
        rows.append({
            "start": event.start.date(), "name": event.name, "date_basis": event.date_basis,
            "alert_before_start": bool(before.fillna(False).any()),
            "alert_by_end": bool(through_event.fillna(False).any()),
            "days_scored": int(through_event.notna().sum()),
        })
    return pd.DataFrame(rows)


def alert_episodes(alerts: pd.Series, labels: pd.DataFrame) -> dict:
    """Group consecutive alert days and count those with no reported flood nearby."""
    alert_days = alerts[alerts.fillna(False).astype(bool)].index
    if len(alert_days) == 0:
        return {"alert_days": 0, "episodes": 0, "false_episodes": 0}
    groups = np.split(alert_days, np.where(np.diff(alert_days) > pd.Timedelta(days=2))[0] + 1)
    false_episodes = 0
    for group in groups:
        window = labels.loc[group[0] - pd.Timedelta(days=3):group[-1] + pd.Timedelta(days=4)]
        if not window["flood_day"].any():
            false_episodes += 1
    return {"alert_days": len(alert_days), "episodes": len(groups),
            "false_episodes": false_episodes}


def wet_season_equivalents(index: pd.DatetimeIndex) -> float:
    """Convert scored June-November days into equivalent full wet seasons."""
    return float(index.month.isin(WET_MONTHS).sum()) / WET_SEASON_DAYS


def evaluate(name, y, score, alerts, catalog, labels, evaluation) -> tuple[dict, pd.DataFrame]:
    predicted = alerts.loc[y.index].astype(bool).to_numpy()
    actual = y.to_numpy()
    hits = int(np.sum(predicted & actual))
    events = score_events(alerts, catalog)
    documented = events[events["date_basis"] == "documented"]
    detected = int(events["alert_by_end"].sum())
    low, high = wilson_interval(detected, len(events))
    # Years with masked wet seasons cannot produce wet-season alerts, so rates
    # use the equivalent number of scored June-November seasons.
    seasons = wet_season_equivalents(alerts.index)
    episodes = alert_episodes(alerts, labels)
    return {
        "model": name, "evaluation": evaluation,
        "days": len(y), "flood_target_days": int(actual.sum()),
        "pr_auc": round(average_precision_score(actual, score), 3),
        "roc_auc": round(roc_auc_score(actual, score), 3),
        "precision": round(hits / max(predicted.sum(), 1), 3),
        "recall": round(hits / max(actual.sum(), 1), 3),
        "events": len(events), "events_detected": detected,
        "event_recall_ci_low": round(low, 2), "event_recall_ci_high": round(high, 2),
        "documented_events": len(documented),
        "documented_warned_before_start": int(documented["alert_before_start"].sum()),
        "false_alert_episodes_per_year": round(episodes["false_episodes"] / max(seasons, 1), 1),
        "alert_days_per_year": round(episodes["alert_days"] / max(seasons, 1), 1),
    }, events


def leave_one_year_out(factory, X, y, swap_X=None, swap_years=()) -> tuple[pd.Series, pd.Series]:
    """Score each year with a model that never saw that year."""
    scores, alerts = [], []
    for year in sorted(set(X.index.year)):
        train = X.index.year != year
        model, threshold = fit_with_threshold(factory, X[train], y[train])
        test_X = swap_X[swap_X.index.year == year] if year in swap_years else X[~train]
        score = pd.Series(model.predict_proba(test_X)[:, 1], index=test_X.index)
        scores.append(score)
        alerts.append(score >= threshold)
    return pd.concat(scores).sort_index(), pd.concat(alerts).sort_index()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
    features = build_features(daily).dropna()
    labels = build_labels(features.index, catalog)

    usable = labels["usable"]
    X_all = features[usable]
    y_all = labels.loc[usable, "target"].astype(int)
    dev = X_all.index.year < HOLDOUT_YEAR
    X_dev, y_dev = X_all[dev], y_all[dev]
    X_hold, y_hold = X_all[~dev], y_all[~dev]

    print(f"Usable days {len(X_all):,}; masked {int((~usable).sum()):,}")
    print(f"Development days {len(X_dev):,} with {int(y_dev.sum())} flood-target days")
    print(f"{HOLDOUT_YEAR} holdout days {len(X_hold):,} with {int(y_hold.sum())} flood-target days")

    results, event_tables = [], []
    for name, factory in model_factories().items():
        score, alerts = leave_one_year_out(factory, X_dev, y_dev)
        row, events = evaluate(name, y_dev, score.loc[y_dev.index], alerts, catalog, labels,
                               "leave_one_year_out_2011_2025")
        results.append(row)
        event_tables.append(events.assign(model=name, evaluation=row["evaluation"]))
        print(f"{name:24s} PR-AUC {row['pr_auc']:.3f} events "
              f"{row['events_detected']}/{row['events']} false/yr "
              f"{row['false_alert_episodes_per_year']}")

    # Model choice uses development years only, so 2026 stays an unseen test.
    candidates = ["lightgbm", "logistic_regression"]
    chosen = max((r for r in results if r["model"] in candidates), key=lambda r: r["pr_auc"])["model"]
    print(f"Selected from development years: {chosen}")

    for name in dict.fromkeys([chosen, "rule_3day_rain", "rule_river_discharge"]):
        factory = model_factories()[name]
        model, threshold = fit_with_threshold(factory, X_dev, y_dev)
        score = pd.Series(model.predict_proba(X_hold)[:, 1], index=X_hold.index)
        row, events = evaluate(name, y_hold, score, score >= threshold, catalog, labels,
                               f"holdout_{HOLDOUT_YEAR}")
        results.append(row)
        event_tables.append(events.assign(model=name, evaluation=row["evaluation"]))

    # Train on ERA5, then test on inputs resembling those a live system receives.
    if LIVE_ARCHIVE_FILE.exists():
        live_daily = pd.read_csv(LIVE_ARCHIVE_FILE, parse_dates=["date"], index_col="date")
        live_features = build_features(live_daily).dropna()
        live_labels = build_labels(live_features.index, catalog)
        live_X = live_features[live_labels["usable"]]
        live_X = live_X[live_X.index.year.isin(LIVE_ARCHIVE_YEARS)]
        live_y = live_labels.loc[live_X.index, "target"].astype(int)
        scores, alerts = [], []
        for year in LIVE_ARCHIVE_YEARS:
            train = X_all.index.year != year
            model, threshold = fit_with_threshold(model_factories()[chosen], X_all[train], y_all[train])
            test = live_X[live_X.index.year == year]
            score = pd.Series(model.predict_proba(test)[:, 1], index=test.index)
            scores.append(score)
            alerts.append(score >= threshold)
        row, events = evaluate(chosen, live_y, pd.concat(scores), pd.concat(alerts), catalog,
                               labels, "live_like_inputs_2023_2026")
        results.append(row)
        event_tables.append(events.assign(model=chosen, evaluation=row["evaluation"]))

    metrics = pd.DataFrame(results)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    pd.concat(event_tables).to_csv(REPORT_DIR / "event_detection.csv", index=False)
    print(metrics.drop(columns=["days"]).T.to_string())
    print("Saved reports only. improve.py selects and saves the model.")


if __name__ == "__main__":
    main()
