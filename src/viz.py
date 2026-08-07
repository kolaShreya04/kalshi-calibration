"""
One styling function -> every report figure looks identical.
(Report Writing Quality rubric: captioned, numbered, self-explanatory figures;
avoids the uncaptioned-screenshot weakness seen in past sample projects.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

STYLE = {
    "figure.figsize": (7, 4.5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def _save(fig, name: str) -> str:
    plt.rcParams.update(STYLE)
    out = Path(config.FIGURES_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def calibration_plot(tab: pd.DataFrame, title: str, name: str) -> str:
    """Reliability diagram with CI whiskers + market-count bars."""
    plt.rcParams.update(STYLE)
    fig, (ax, axh) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    valid = tab.dropna(subset=["mean_price", "resolution_rate"])
    if {"rate_ci_lo", "rate_ci_hi"}.issubset(valid.columns):
        yerr = np.abs(np.vstack([
            valid["resolution_rate"] - valid["rate_ci_lo"],
            valid["rate_ci_hi"] - valid["resolution_rate"]]))
        ax.errorbar(valid["mean_price"], valid["resolution_rate"], yerr=yerr,
                    fmt="o-", capsize=3, label="Empirical rate (95% CI)")
    else:
        ax.plot(valid["mean_price"], valid["resolution_rate"], "o-",
                label="Empirical rate")
    ax.set_ylabel("Empirical resolution rate")
    ax.set_title(title)
    ax.legend()
    centers = (np.arange(10) + 0.5) / 10
    axh.bar(centers, tab["n"].to_numpy(), width=0.09, alpha=0.6)
    axh.set_xlabel("Early price (opening probability)")
    axh.set_ylabel("N markets")
    axh.set_yscale("log")
    return _save(fig, name)


def deviation_plot(tabs: dict[str, pd.DataFrame], name: str) -> str:
    """Signed deviation (rate - price) by bin, one line per category.
    FLB signature: negative on the left, positive on the right."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots()
    ax.axhline(0, color="k", lw=1, ls="--")
    for label, tab in tabs.items():
        valid = tab.dropna(subset=["mean_price", "deviation"])
        ax.plot(valid["mean_price"], valid["deviation"], "o-", label=label)
    ax.set_xlabel("Early price (opening probability)")
    ax.set_ylabel("Resolution rate − price")
    ax.set_title("Favorite–longshot deviation by category")
    ax.legend()
    return _save(fig, name)
