"""
Workstream 4: open-to-resolution drift and tradeability (proposal §3.3).

Drift: a market's price at resolution IS its settlement value (0/1), so the
open-to-resolution drift equals the residual  y - p. We report mean drift
per (early-price bin × category) with event-cluster bootstrap CIs.

Tradeability backtest — designed so a NULL result is a clean finding:
1. TEMPORAL split: rules are learned on the earliest TRAIN_FRAC of markets
   (by close_time) and evaluated strictly out-of-sample on the remainder.
   No rule ever sees its own evaluation data (proposal §4: no leakage).
2. Rule: for each (category × price bin) cell whose train-period drift CI
   excludes zero, trade in the drift's direction at open in the test period.
3. Execution realism (the peer-review-praised part):
   - entry at the quoted side of the book: buy YES at mid + spread/2,
     buy NO at 1 - (mid - spread/2); spread from opening-minute quotes.
   - Kalshi trading fee: fee = FEE_RATE * price * (1 - price) per contract
     (Kalshi fee schedule; general markets use 0.07 — cite in report).
4. Report gross AND net edge per contract with cluster-bootstrap CIs.
   Net edge ≤ 0 with tight CIs = "prices imperfect but not exploitable
   after costs" — exactly the hypothesis the proposal commits to testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import calibration

FEE_RATE = 0.07          # Kalshi general fee multiplier (cite fee schedule)
TRAIN_FRAC = 0.7
MIN_CELL_N = 50          # min train markets for a cell to be eligible
CI_LEVEL = 0.95


def _prep(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    d = calibration._prep(df, price_col)
    extra = df[["ticker", "close_time", "open_spread"]].copy()
    d = d.merge(extra, on="ticker", how="left")
    d["bin"] = pd.cut(d["p"], calibration.BIN_EDGES, include_lowest=True)
    d["drift"] = d["y"] - d["p"]
    return d.sort_values("close_time").reset_index(drop=True)


def drift_table(df: pd.DataFrame, price_col: str, n_boot: int = 1000
                ) -> pd.DataFrame:
    """Mean drift per category × bin with cluster-bootstrap CIs."""
    d = _prep(df, price_col)
    rows = []
    for (cat, b), g in d.groupby(["category", "bin"], observed=True):
        if len(g) < MIN_CELL_N:
            continue
        boots = calibration.cluster_bootstrap(
            g, lambda x: x["drift"].mean(), n_boot=n_boot)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"category": cat, "bin": str(b), "n": len(g),
                     "mean_drift": g["drift"].mean(),
                     "ci_lo": lo, "ci_hi": hi,
                     "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def kalshi_fee(price: np.ndarray) -> np.ndarray:
    """Per-contract trading fee in dollars (general markets)."""
    return FEE_RATE * price * (1 - price)


def backtest(df: pd.DataFrame, price_col: str, n_boot: int = 1000,
             use_spread: bool = True
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward rule backtest. Returns (per_trade, summary).

    use_spread=True  charges half the OPENING spread on entry — a
        conservative UPPER bound on execution cost (the book at the first
        trade is typically tighter than at open).
    use_spread=False charges fees only — a LOWER bound (assumes execution
        at the first-trade price itself). True cost lies between.
    """
    d = _prep(df, price_col)
    split = int(len(d) * TRAIN_FRAC)
    train, test = d.iloc[:split], d.iloc[split:]

    # learn signal cells on the prepped train frame
    rows = []
    for (cat, b), g in train.groupby(["category", "bin"], observed=True):
        if len(g) < MIN_CELL_N:
            continue
        boots = calibration.cluster_bootstrap(
            g, lambda x: x["drift"].mean(), n_boot=n_boot)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        if lo > 0 or hi < 0:
            rows.append({"category": cat, "bin": b,
                         "direction": 1 if g["drift"].mean() > 0 else -1,
                         "train_drift": g["drift"].mean(), "train_n": len(g)})
    signals = pd.DataFrame(rows)
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "price_def": price_col, "n_signal_cells": 0, "n_trades": 0,
            "note": "no significant drift cells on train — null result"}])

    trades = test.merge(signals, on=["category", "bin"], how="inner")
    if use_spread:
        half_spread = (trades["open_spread"]
                       .fillna(trades["open_spread"].median()).fillna(0.02) / 2)
    else:
        half_spread = pd.Series(0.0, index=trades.index)
    # direction +1: buy YES at p + half_spread, payoff y
    # direction -1: buy NO  at (1 - p) + half_spread, payoff 1 - y
    entry = np.where(trades["direction"] == 1,
                     trades["p"] + half_spread,
                     (1 - trades["p"]) + half_spread)
    payoff = np.where(trades["direction"] == 1, trades["y"], 1 - trades["y"])
    trades["gross_pnl"] = payoff - entry
    trades["fee"] = kalshi_fee(np.clip(entry, 0.01, 0.99))
    trades["net_pnl"] = trades["gross_pnl"] - trades["fee"]

    def _sum(rowset: pd.DataFrame, scope: str) -> dict:
        g_boots = calibration.cluster_bootstrap(
            rowset, lambda x: x["gross_pnl"].mean(), n_boot=n_boot)
        n_boots = calibration.cluster_bootstrap(
            rowset, lambda x: x["net_pnl"].mean(), n_boot=n_boot)
        return {
            "price_def": price_col, "scope": scope,
            "n_signal_cells": len(signals), "n_trades": len(rowset),
            "gross_edge": rowset["gross_pnl"].mean(),
            "gross_ci_lo": np.percentile(g_boots, 2.5),
            "gross_ci_hi": np.percentile(g_boots, 97.5),
            "net_edge": rowset["net_pnl"].mean(),
            "net_ci_lo": np.percentile(n_boots, 2.5),
            "net_ci_hi": np.percentile(n_boots, 97.5),
            "hit_rate": (rowset["net_pnl"] > 0).mean(),
        }

    summaries = [_sum(trades, "ALL")]
    summaries += [_sum(g, cat) for cat, g in trades.groupby("category")
                  if len(g) >= 30]
    return trades, pd.DataFrame(summaries)
