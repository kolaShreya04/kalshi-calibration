"""
M4 entry point: drift + tradeability.

Prereq: data/processed/early_prices.parquet

Outputs:
  data/processed/m4_drift_table.csv     drift per category × price bin (+CIs)
  data/processed/m4_trades.csv          out-of-sample simulated trades
  data/processed/m4_backtest_summary.csv gross vs net edge (+CIs)
  figures/m4_drift_heatmap.png
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import drift, early_price  # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(f"{config.PROCESSED_DIR}/early_prices.parquet")
    df = df[early_price.usable_mask(df)]
    price_col = config.PRIMARY_PRICE_COL

    out = Path(config.PROCESSED_DIR)
    tab = drift.drift_table(df, price_col, n_boot=args.n_boot)
    tab.to_csv(out / "m4_drift_table.csv", index=False)

    trades, summary = drift.backtest(df, price_col, n_boot=args.n_boot)
    if not trades.empty:
        keep = ["ticker", "category", "bin", "direction", "p", "y",
                "gross_pnl", "fee", "net_pnl"]
        trades[[c for c in keep if c in trades.columns]].to_csv(
            out / "m4_trades.csv", index=False)
    summary.to_csv(out / "m4_backtest_summary.csv", index=False)

    # drift heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.viz import STYLE, _save
        plt.rcParams.update(STYLE)
        pivot = tab.pivot_table(index="category", columns="bin",
                                values="mean_drift", observed=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-0.15, vmax=0.15,
                       aspect="auto")
        ax.set_xticks(range(len(pivot.columns)),
                      [str(c) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        fig.colorbar(im, label="Mean drift (resolution − early price)")
        ax.set_title("Open-to-resolution drift by category and price bin")
        _save(fig, "m4_drift_heatmap")
    except Exception as e:  # noqa: BLE001
        log.warning("heatmap failed: %s", e)

    print("\n=== M4 backtest summary ===")
    print(summary.to_string(index=False))
    print("\nReminder: net edge ≤ 0 with tight CIs is a *finding* — opening"
          "\nprices imperfect but not exploitable after fees and spread.")


if __name__ == "__main__":
    main()
