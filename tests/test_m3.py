"""
M3 validation on synthetic data with KNOWN structure.
Run: python tests/test_m3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import residual_model


def make_synth(n=6000, crypto_bias=0.0, seed=0) -> pd.DataFrame:
    """Markets priced p; Crypto markets' TRUE prob is p - crypto_bias
    (overpriced) — an observable-feature-driven mispricing."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    cat = rng.choice(["Sports", "Politics", "Crypto"], n)
    true_prob = np.where(cat == "Crypto", np.clip(p - crypto_bias, 0.01, 0.99), p)
    y = rng.binomial(1, true_prob)
    open_t = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        np.sort(rng.uniform(0, 500, n)), unit="D")
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "event_ticker": [f"E{i//3}" for i in range(n)],
        "category": cat,
        "result": np.where(y == 1, "yes", "no"),
        "price": p,
        "open_time": open_t,
        "close_time": open_t + pd.Timedelta(days=2),
        "open_interest": rng.exponential(1000, n),
        "life10_volume": rng.exponential(200, n),
        "open_spread": rng.uniform(0.01, 0.05, n),
        "first_trade_frac": rng.uniform(0, 0.25, n),
    })


def test_features_sorted_by_close_time():
    d = residual_model.build_features(make_synth(n=500), "price")
    ct = pd.to_datetime(d["close_time"], utc=True)
    assert ct.is_monotonic_increasing, "temporal CV requires time ordering"


def test_no_gain_when_market_efficient():
    df = make_synth(n=6000, crypto_bias=0.0, seed=1)
    folds, _ = residual_model.evaluate(df, "price", n_splits=3)
    s = residual_model.summarize_folds(folds)
    gain = s[s["model"] == "grad_boosting"]["auc_gain_vs_price_only"].iloc[0]
    assert abs(gain) < 0.02, f"no real structure but AUC gain = {gain:.3f}"


def test_detects_feature_driven_mispricing():
    df = make_synth(n=6000, crypto_bias=0.15, seed=2)
    folds, imps = residual_model.evaluate(df, "price", n_splits=3)
    s = residual_model.summarize_folds(folds)
    feature_models = s[s["model"] != "price_only_logistic"]
    best_gain = feature_models["ll_gain_vs_price_only"].max()
    assert best_gain > 0.005, (
        f"best feature model should beat price-only baseline, gain={best_gain:.4f}")
    assert not imps.empty
    top = imps.iloc[0]["feature"]
    assert top in ("category", "logit_p"), f"category should drive importance, got {top}"


def test_fold_metrics_complete():
    df = make_synth(n=3000, crypto_bias=0.1, seed=3)
    folds, _ = residual_model.evaluate(df, "price", n_splits=3)
    assert set(folds["model"]) == {"price_only_logistic", "logistic_l2",
                                   "random_forest", "grad_boosting"}
    assert folds["auc"].between(0, 1).all()
    assert (folds["n_train"] > 0).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
