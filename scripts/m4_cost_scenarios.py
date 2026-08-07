"""
M4 cost-scenario bounds: fee-only (lower bound on cost) vs. fee + opening
half-spread (conservative upper bound). The true execution cost of trading
at the first-trade time lies between the two; reporting both brackets the
answer honestly (report Results section, cost-sensitivity paragraph).

Usage: python scripts/m4_cost_scenarios.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import drift, early_price  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    df = pd.read_parquet(f"{config.PROCESSED_DIR}/early_prices.parquet")
    df = df[early_price.usable_mask(df)]
    price_col = config.PRIMARY_PRICE_COL

    frames = []
    for label, use_spread in [("fees_only (lower bound)", False),
                              ("fees+open_spread (upper bound)", True)]:
        _, summary = drift.backtest(df, price_col, n_boot=1000,
                                    use_spread=use_spread)
        summary.insert(0, "cost_scenario", label)
        frames.append(summary)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(f"{config.PROCESSED_DIR}/m4_cost_scenarios.csv", index=False)

    cols = ["cost_scenario", "scope", "n_trades", "gross_edge",
            "net_edge", "net_ci_lo", "net_ci_hi", "hit_rate"]
    print("\n=== M4 net edge under bounding cost scenarios ===")
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    print("\nRead: if net_ci_hi < 0 in BOTH scenarios, no exploitable edge"
          "\nexists under any realistic cost assumption; if the fee-only"
          "\nscenario shows positive net edge, the edge exists in principle"
          "\nbut is destroyed by the spread.")


if __name__ == "__main__":
    main()
