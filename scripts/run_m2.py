"""
M2 entry point: calibration + favorite-longshot analysis.

Prereq: data/processed/early_prices.parquet (python src/collect_early.py).

Runs every analysis under BOTH liquidity-filtered and unfiltered samples,
and under all early-price definitions (peer-review robustness axis).

Outputs:
  data/processed/calibration_summary.csv
  data/processed/calibration_bins_<def>_<scope>.csv
  data/processed/flb_summary.csv
  figures/calibration_<def>_<scope>.png
  figures/flb_deviation_<def>.png
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import calibration, early_price, flb, viz  # noqa: E402

log = logging.getLogger(__name__)

PRICE_DEFS = ["first_trade_price", "first_day_vwap", "life10_vwap"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--no-liquidity-filter", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(f"{config.PROCESSED_DIR}/early_prices.parquet")
    n0 = len(df)
    if not args.no_liquidity_filter:
        df = df[early_price.usable_mask(df)]
        log.info("usable-market filter: %d -> %d markets", n0, len(df))

    out = Path(config.PROCESSED_DIR)
    cal_summaries, flb_summaries = [], []

    for price_col in PRICE_DEFS:
        if price_col not in df.columns or df[price_col].notna().sum() < 100:
            log.warning("skipping %s (insufficient data)", price_col)
            continue
        log.info("=== price definition: %s ===", price_col)
        cal_summaries.append(calibration.summarize(df, price_col,
                                                   n_boot=args.n_boot))
        flb_summaries.append(flb.analyze(df, price_col, n_boot=args.n_boot))

        d = calibration._prep(df, price_col)
        tabs = {}
        for scope, g in [("ALL", d)] + list(d.groupby("category")):
            if len(g) < 100:
                continue
            tab = calibration.bin_rate_cis(g, n_boot=min(args.n_boot, 1000))
            tab.to_csv(out / f"calibration_bins_{price_col}_{scope}.csv",
                       index=False)
            tabs[scope] = tab
            viz.calibration_plot(
                tab, f"Kalshi opening-price calibration — {scope} ({price_col})",
                f"calibration_{price_col}_{scope}")
        if tabs:
            viz.deviation_plot(
                {k: v for k, v in tabs.items() if k != "ALL"} or tabs,
                f"flb_deviation_{price_col}")

    pd.concat(cal_summaries, ignore_index=True).to_csv(
        out / "calibration_summary.csv", index=False)
    pd.concat(flb_summaries, ignore_index=True).to_csv(
        out / "flb_summary.csv", index=False)

    print("\n=== Calibration headline (primary definition) ===")
    cal = pd.concat(cal_summaries, ignore_index=True)
    primary = cal[cal["price_def"] == config.PRIMARY_PRICE_COL]
    cols = ["scope", "n_markets", "brier", "brier_baseline", "ece",
            "reliability", "resolution"]
    print(primary[cols].to_string(index=False))
    print(f"\nAll tables in {out}/, figures in {config.FIGURES_DIR}/")



if __name__ == "__main__":
    main()
