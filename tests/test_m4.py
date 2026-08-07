"""
M4 validation on synthetic data with KNOWN drift structure.
Run: python tests/test_m4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import drift


def make_synth(n=8000, biased_cell=False, seed=0) -> pd.DataFrame:
    """If biased_cell: Sports markets priced in [0.6, 0.8] resolve YES 12pp
    more often than priced (persistent positive drift); all else calibrated."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    cat = rng.choice(["Sports", "Politics"], n)
    true_prob = p.copy()
    if biased_cell:
        mask = (cat == "Sports") & (p >= 0.6) & (p < 0.8)
        true_prob[mask] = np.clip(p[mask] + 0.12, 0, 0.99)
    y = rng.binomial(1, true_prob)
    close_t = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        np.sort(rng.uniform(0, 500, n)), unit="D")
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "event_ticker": [f"E{i//2}" for i in range(n)],
        "category": cat, "result": np.where(y == 1, "yes", "no"),
        "price": p, "close_time": close_t,
        "open_spread": rng.uniform(0.01, 0.03, n),
    })


def test_fee_formula():
    fees = drift.kalshi_fee(np.array([0.5, 0.1, 0.9]))
    assert abs(fees[0] - 0.07 * 0.25) < 1e-12
    assert abs(fees[1] - fees[2]) < 1e-12  # symmetric


def test_drift_table_flags_biased_cell():
    tab = drift.drift_table(make_synth(biased_cell=True, seed=1), "price",
                            n_boot=300)
    sig = tab[tab["significant"]]
    assert not sig.empty
    top = sig.loc[sig["mean_drift"].abs().idxmax()]
    assert top["category"] == "Sports" and "0.6" in top["bin"] or "0.7" in top["bin"]
    assert top["mean_drift"] > 0.06


def test_backtest_finds_edge_when_it_exists():
    trades, summary = drift.backtest(make_synth(biased_cell=True, seed=2),
                                     "price", n_boot=300)
    assert not trades.empty
    row = summary[summary["scope"] == "ALL"].iloc[0]
    assert row["gross_edge"] > 0.02, f"gross edge {row['gross_edge']:.4f}"
    assert row["net_edge"] < row["gross_edge"], "fees+spread must bite"


def test_backtest_null_when_efficient():
    trades, summary = drift.backtest(make_synth(biased_cell=False, seed=3),
                                     "price", n_boot=300)
    if trades.empty:
        assert summary.iloc[0]["n_trades"] == 0  # clean null
    else:
        # few false-positive cells may survive; net edge must be ~0 or worse
        row = summary[summary["scope"] == "ALL"].iloc[0]
        assert row["net_edge"] < 0.02


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
