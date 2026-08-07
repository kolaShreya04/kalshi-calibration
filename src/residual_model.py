"""
Workstream 3 (primary ML anchor): predicting where mispricing concentrates.

Framing (proposal §3.2, refined per peer feedback "specify which models"):
Under efficiency, the outcome y is unpredictable given the opening price p
beyond p itself. We therefore compare

  BASELINE  price-only logistic:  y ~ sigmoid(a + b*logit(p))
            (a recalibration of the market price — the strongest fair
            baseline; beating raw p alone would be too easy if prices
            are miscalibrated in aggregate)
  MODELS    L2 logistic regression / random forest / gradient boosting
            on p PLUS market features: category, early volume, open
            interest, time-to-resolution, opening spread, time of first
            trade, open hour/day-of-week.

If the feature models beat the price-only baseline on held-out AUC and
log loss, mispricing is SYSTEMATIC and attributable (via permutation
importance) to observable market characteristics.

Validation: expanding-window TEMPORAL cross-validation ordered by
close_time — a model is always evaluated on markets that resolve strictly
after everything it trained on (no temporal leakage; proposal §4).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

log = logging.getLogger(__name__)

NUMERIC = ["logit_p", "log_early_volume", "log_open_interest",
           "log_hours_to_resolution", "open_spread", "first_trade_frac",
           "open_hour"]
CATEGORICAL = ["category", "open_dow"]


def build_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Feature table; one row per market, ordered by close_time."""
    d = df.copy()
    d = d.dropna(subset=[price_col, "result"])
    d = d[(d[price_col] > 0.01) & (d[price_col] < 0.99)]
    d["y"] = (d["result"] == "yes").astype(int)
    d["p"] = d[price_col]
    d["logit_p"] = np.log(d["p"] / (1 - d["p"]))

    open_t = pd.to_datetime(d["open_time"], utc=True)
    close_t = pd.to_datetime(d["close_time"], utc=True)
    d["log_hours_to_resolution"] = np.log1p(
        (close_t - open_t).dt.total_seconds() / 3600.0)
    d["open_hour"] = open_t.dt.hour
    d["open_dow"] = open_t.dt.dayofweek.astype(str)

    d["log_early_volume"] = np.log1p(
        d.get("life10_volume", pd.Series(0, index=d.index)).fillna(0))
    d["log_open_interest"] = np.log1p(d["open_interest"].fillna(0))
    d["open_spread"] = d["open_spread"].fillna(d["open_spread"].median())
    d["first_trade_frac"] = d["first_trade_frac"].fillna(
        d["first_trade_frac"].median())

    return d.sort_values("close_time").reset_index(drop=True)


def _models() -> dict:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return {
        "price_only_logistic": Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000))]),  # fed logit_p only
        "logistic_l2": Pipeline([
            ("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=1000))]),
        "random_forest": Pipeline([
            ("pre", pre), ("rf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                random_state=config.CENSUS_RANDOM_SEED))]),
        "grad_boosting": Pipeline([
            ("pre", pre), ("gb", HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, l2_regularization=1.0,
                random_state=config.CENSUS_RANDOM_SEED))]),
    }


def evaluate(df: pd.DataFrame, price_col: str, n_splits: int = 5
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window temporal CV. Returns (fold_metrics, importances)."""
    d = build_features(df, price_col)
    X_full = d[NUMERIC + CATEGORICAL]
    y = d["y"].to_numpy()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows, imp_rows = [], []

    for fold, (tr, te) in enumerate(tscv.split(d)):
        if y[tr].std() == 0 or y[te].std() == 0:
            continue
        for name, model in _models().items():
            if name == "price_only_logistic":
                Xtr, Xte = d[["logit_p"]].iloc[tr], d[["logit_p"]].iloc[te]
            else:
                Xtr, Xte = X_full.iloc[tr], X_full.iloc[te]
            model.fit(Xtr, y[tr])
            prob = model.predict_proba(Xte)[:, 1]
            rows.append({
                "fold": fold, "model": name, "price_def": price_col,
                "n_train": len(tr), "n_test": len(te),
                "auc": roc_auc_score(y[te], prob),
                "log_loss": log_loss(y[te], prob, labels=[0, 1]),
                "brier": brier_score_loss(y[te], prob),
                "market_brier": brier_score_loss(y[te], d["p"].iloc[te]),
            })
            # permutation importance on the last (largest-train) fold only
            if name == "grad_boosting" and fold == n_splits - 1:
                pi = permutation_importance(
                    model, Xte, y[te], n_repeats=10,
                    random_state=config.CENSUS_RANDOM_SEED,
                    scoring="neg_log_loss")
                for feat, mean, sd in zip(X_full.columns,
                                          pi.importances_mean,
                                          pi.importances_std):
                    imp_rows.append({"feature": feat, "importance": mean,
                                     "importance_sd": sd,
                                     "price_def": price_col})

    folds = pd.DataFrame(rows)
    importances = (pd.DataFrame(imp_rows)
                   .sort_values("importance", ascending=False)
                   if imp_rows else pd.DataFrame())
    return folds, importances


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean ± sd across folds; the report's model-comparison table."""
    g = (folds.groupby(["price_def", "model"])
         .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
              log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"),
              brier_mean=("brier", "mean"),
              market_brier_mean=("market_brier", "mean"),
              n_folds=("fold", "nunique"))
         .reset_index())
    # gain vs the price-only baseline within each price definition
    base = g[g["model"] == "price_only_logistic"][["price_def", "auc_mean", "log_loss_mean"]]
    base = base.rename(columns={"auc_mean": "base_auc", "log_loss_mean": "base_ll"})
    g = g.merge(base, on="price_def")
    g["auc_gain_vs_price_only"] = g["auc_mean"] - g["base_auc"]
    g["ll_gain_vs_price_only"] = g["base_ll"] - g["log_loss_mean"]
    return g.drop(columns=["base_auc", "base_ll"])
