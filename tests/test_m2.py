"""
M2 statistical validation on synthetic data with KNOWN ground truth.
Run: python tests/test_m2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import calibration, flb


def make_synth(n=8000, logit_slope=1.0, seed=0) -> pd.DataFrame:
    """Markets whose true P(yes) = sigmoid(logit_slope * logit(price)).
    slope=1 -> perfectly calibrated; slope>1 -> favorite-longshot bias."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    x = np.log(p / (1 - p))
    true_prob = 1 / (1 + np.exp(-logit_slope * x))
    y = rng.binomial(1, true_prob)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "event_ticker": [f"E{i // 4}" for i in range(n)],  # 4 markets/event
        "category": rng.choice(["Sports", "Politics"], n),
        "result": np.where(y == 1, "yes", "no"),
        "price": p,
    })


def test_calibrated_market_metrics():
    d = calibration._prep(make_synth(logit_slope=1.0), "price")
    assert calibration.ece(d) < 0.02, "ECE should be near 0 when calibrated"
    b = calibration.brier(d)
    expected = (d["p"] * (1 - d["p"])).mean()  # E[Brier] under calibration
    assert abs(b - expected) < 0.01
    dec = calibration.brier_decomposition(d)
    assert dec["reliability"] < 0.001
    # Murphy identity holds up to within-bin variance term
    assert abs(dec["brier_check"] - b) < 0.01


def test_flb_null_and_detection():
    calm = calibration._prep(make_synth(logit_slope=1.0, seed=1), "price")
    assert abs(flb.flb_stat(calm)) < 0.03, "no FLB when calibrated"
    _, b_hat = flb.logit_slope(calm)
    assert abs(b_hat - 1.0) < 0.08, f"slope ~1 expected, got {b_hat}"

    biased = calibration._prep(make_synth(logit_slope=1.6, seed=2), "price")
    stat = flb.flb_stat(biased)
    assert stat > 0.03, f"FLB stat should be clearly positive, got {stat}"
    _, b_hat = flb.logit_slope(biased)
    assert abs(b_hat - 1.6) < 0.15, f"slope ~1.6 expected, got {b_hat}"


def test_cluster_bootstrap_ci_covers_truth():
    d = calibration._prep(make_synth(logit_slope=1.0, seed=3, n=4000), "price")
    boots = calibration.cluster_bootstrap(d, calibration.ece, n_boot=200)
    assert boots.shape == (200,)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    assert lo <= calibration.ece(d) <= hi


def test_cluster_bootstrap_wider_than_iid_under_correlation():
    """Perfectly correlated outcomes within events -> cluster CI must be wider."""
    rng = np.random.default_rng(4)
    n_events = 300
    rows = []
    for e in range(n_events):
        p = rng.uniform(0.3, 0.7)
        y = rng.binomial(1, p)          # SAME outcome for all 5 markets
        for k in range(5):
            rows.append({"ticker": f"T{e}_{k}", "event_ticker": f"E{e}",
                         "category": "Sports",
                         "result": "yes" if y else "no", "price": p})
    df = calibration._prep(pd.DataFrame(rows), "price")
    mean_y = lambda d: d["y"].mean()
    cluster_sd = calibration.cluster_bootstrap(df, mean_y, n_boot=300).std()
    iid = df.copy()
    iid["event_ticker"] = np.arange(len(iid)).astype(str)  # break clustering
    iid_sd = calibration.cluster_bootstrap(iid, mean_y, n_boot=300).std()
    assert cluster_sd > 1.5 * iid_sd, (
        f"cluster SD {cluster_sd:.4f} should exceed iid SD {iid_sd:.4f}")


def test_summarize_and_analyze_run_end_to_end():
    df = make_synth(n=3000, logit_slope=1.4, seed=5)
    cal = calibration.summarize(df, "price", n_boot=100)
    assert {"ALL", "Sports", "Politics"} == set(cal["scope"])
    assert (cal["brier"] < cal["brier_baseline"]).all(), \
        "market prices should beat the no-skill baseline"
    f = flb.analyze(df, "price", n_boot=100)
    assert (f[f["scope"] == "ALL"]["flb_stat"] > 0).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")

