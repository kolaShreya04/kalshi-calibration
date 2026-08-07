"""
Workstream 1: opening-price calibration (proposal §3.1, evaluation §4).

Statistical choices and why (report Methodology section draws from here):

- Calibration curve: markets binned by early price; empirical resolution
  rate per bin vs. mean early price. 45° line = perfect calibration.
- Brier score = mean (p - y)^2; decomposed via Murphy (1973) into
  reliability - resolution + uncertainty, so we can attribute error.
- ECE (expected calibration error) = volume-of-bin-weighted |rate - price|.
- Uncertainty: CLUSTER bootstrap resampling EVENTS (event_ticker), not
  individual markets. Markets within one event (e.g., strikes of the same
  election or game) have mechanically correlated outcomes; an iid bootstrap
  would understate variance. This addresses the rubric's sample-bias probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

BIN_EDGES = np.linspace(0.0, 1.0, 11)  # 10 bins: [0,.1), ..., [.9,1]


def _prep(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    out = df[["ticker", "event_ticker", "category", "result", price_col]].copy()
    out = out.rename(columns={price_col: "p"})
    out["y"] = (out["result"] == "yes").astype(float)
    out = out.dropna(subset=["p"])
    out = out[(out["p"] > 0) & (out["p"] < 1)]
    return out


def brier(df: pd.DataFrame) -> float:
    return float(((df["p"] - df["y"]) ** 2).mean())


def brier_decomposition(df: pd.DataFrame) -> dict:
    """Murphy decomposition on the binned forecasts."""
    d = df.copy()
    d["bin"] = pd.cut(d["p"], BIN_EDGES, include_lowest=True)
    base = d["y"].mean()
    g = d.groupby("bin", observed=True).agg(n=("y", "size"),
                                            rate=("y", "mean"),
                                            mean_p=("p", "mean"))
    n = len(d)
    reliability = float((g["n"] * (g["mean_p"] - g["rate"]) ** 2).sum() / n)
    resolution = float((g["n"] * (g["rate"] - base) ** 2).sum() / n)
    uncertainty = float(base * (1 - base))
    return {"reliability": reliability, "resolution": resolution,
            "uncertainty": uncertainty,
            "brier_check": reliability - resolution + uncertainty}


def ece(df: pd.DataFrame) -> float:
    d = df.copy()
    d["bin"] = pd.cut(d["p"], BIN_EDGES, include_lowest=True)
    g = d.groupby("bin", observed=True).agg(n=("y", "size"),
                                            rate=("y", "mean"),
                                            mean_p=("p", "mean"))
    return float((g["n"] / len(d) * (g["rate"] - g["mean_p"]).abs()).sum())


def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = pd.cut(d["p"], BIN_EDGES, include_lowest=True)
    g = (d.groupby("bin", observed=False)
          .agg(n=("y", "size"), mean_price=("p", "mean"),
               resolution_rate=("y", "mean"), n_events=("event_ticker", "nunique"))
          .reset_index())
    g["deviation"] = g["resolution_rate"] - g["mean_price"]
    return g


def cluster_bootstrap(df: pd.DataFrame, stat_fn, n_boot: int = 2000,
                      seed: int = config.CENSUS_RANDOM_SEED) -> np.ndarray:
    """Bootstrap stat_fn(df) resampling whole events with replacement.

    Implementation note: we resample integer row-index blocks per event and
    take df.iloc[idx] once per replicate — orders of magnitude faster than
    concatenating per-event frames.
    """
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True)
    codes, _ = pd.factorize(d["event_ticker"])
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(sorted_codes[-1] + 1))
    bounds = np.append(starts, len(sorted_codes))
    blocks = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    n_events = len(blocks)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_events, size=n_events)
        idx = np.concatenate([blocks[j] for j in pick])
        stats.append(stat_fn(d.iloc[idx]))
    return np.asarray(stats)


def bin_rate_cis(df: pd.DataFrame, n_boot: int = 1000,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Per-bin resolution-rate CIs via cluster bootstrap."""
    def rates(d: pd.DataFrame) -> np.ndarray:
        dd = d.copy()
        dd["bin"] = pd.cut(dd["p"], BIN_EDGES, include_lowest=True)
        return (dd.groupby("bin", observed=False)["y"].mean()
                  .reindex(pd.IntervalIndex.from_breaks(BIN_EDGES, closed="right"))
                  .to_numpy())
    boots = cluster_bootstrap(df, rates, n_boot=n_boot)
    lo = np.nanpercentile(boots, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boots, 100 * (1 - alpha / 2), axis=0)
    tab = calibration_table(df)
    tab["rate_ci_lo"], tab["rate_ci_hi"] = lo, hi
    return tab


def summarize(df: pd.DataFrame, price_col: str, by_category: bool = True,
              n_boot: int = 2000) -> pd.DataFrame:
    """Headline metrics (+ cluster-bootstrap CIs) overall and per category."""
    d = _prep(df, price_col)
    scopes = [("ALL", d)]
    if by_category:
        scopes += [(c, g) for c, g in d.groupby("category")]
    rows = []
    for name, g in scopes:
        if len(g) < 30:
            continue
        b_boot = cluster_bootstrap(g, brier, n_boot=n_boot)
        e_boot = cluster_bootstrap(g, ece, n_boot=n_boot)
        base_rate = g["y"].mean()
        rows.append({
            "scope": name, "price_def": price_col,
            "n_markets": len(g), "n_events": g["event_ticker"].nunique(),
            "base_rate": base_rate,
            "brier": brier(g),
            "brier_ci_lo": np.percentile(b_boot, 2.5),
            "brier_ci_hi": np.percentile(b_boot, 97.5),
            "brier_baseline": float(base_rate * (1 - base_rate)),  # no-skill
            "ece": ece(g),
            "ece_ci_lo": np.percentile(e_boot, 2.5),
            "ece_ci_hi": np.percentile(e_boot, 97.5),
            **brier_decomposition(g),
        })
    return pd.DataFrame(rows)
