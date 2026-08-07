"""
Workstream 2: favorite-longshot bias (proposal §3.1, literature: Thaler &
Ziemba 1988; Snowberg & Wolfers 2010).

FLB prediction: longshots (low p) resolve YES less often than their price
implies (overpriced), favorites (high p) more often (underpriced), i.e.
deviation = rate - price is negative at low p and positive at high p.

Tests:
1. Per-bin signed deviation with cluster-bootstrap CIs (from calibration.py).
2. A single-number FLB statistic:
       flb_stat = mean deviation over favorites (p >= .8)
                - mean deviation over longshots (p <= .2)
   > 0 under classic FLB. Cluster-bootstrap CI + two-sided p-value.
3. Logit-slope test: fit y ~ sigmoid(a + b * logit(p)) by Newton's method.
   b > 1 => prices too extreme are 'shrunk' back (consistent with FLB
   in the tails); b = 1, a = 0 => perfect calibration. Cluster-bootstrap CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import calibration


def flb_stat(df: pd.DataFrame, lo: float = 0.2, hi: float = 0.8) -> float:
    longshot = df[df["p"] <= lo]
    favorite = df[df["p"] >= hi]
    if len(longshot) < 5 or len(favorite) < 5:
        return np.nan
    dev_l = (longshot["y"] - longshot["p"]).mean()
    dev_f = (favorite["y"] - favorite["p"]).mean()
    return float(dev_f - dev_l)


def logit_slope(df: pd.DataFrame, max_iter: int = 50) -> tuple[float, float]:
    """MLE of y ~ sigmoid(a + b*logit(p)) via Newton-Raphson. Returns (a, b)."""
    p = df["p"].to_numpy().clip(1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(max_iter):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = mu * (1 - mu)
        grad = X.T @ (y - mu)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def analyze(df: pd.DataFrame, price_col: str, n_boot: int = 2000) -> pd.DataFrame:
    """FLB summary per scope (ALL + each category)."""
    d = calibration._prep(df, price_col)
    scopes = [("ALL", d)] + [(c, g) for c, g in d.groupby("category")]
    rows = []
    for name, g in scopes:
        if len(g) < 100:
            continue
        stat = flb_stat(g)
        boots = calibration.cluster_bootstrap(g, flb_stat, n_boot=n_boot)
        boots = boots[~np.isnan(boots)]
        slope_boots = calibration.cluster_bootstrap(
            g, lambda b: logit_slope(b)[1], n_boot=min(n_boot, 500))
        a, b = logit_slope(g)
        # two-sided bootstrap p-value for flb_stat != 0
        if len(boots):
            p_two = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        else:
            p_two = np.nan
        rows.append({
            "scope": name, "price_def": price_col, "n_markets": len(g),
            "flb_stat": stat,
            "flb_ci_lo": np.percentile(boots, 2.5) if len(boots) else np.nan,
            "flb_ci_hi": np.percentile(boots, 97.5) if len(boots) else np.nan,
            "flb_p_two_sided": p_two,
            "logit_intercept": a, "logit_slope": b,
            "slope_ci_lo": np.percentile(slope_boots, 2.5),
            "slope_ci_hi": np.percentile(slope_boots, 97.5),
        })
    return pd.DataFrame(rows)
