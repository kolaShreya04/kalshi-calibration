"""
M1 deliverable: the data census.

Answers, per category and month:
  - how many resolved binary markets do we have?
  - what fraction have any early-window liquidity (and how much)?
  - what do spreads at open look like?

The summary table goes straight into the final report (Data Cleaning /
sample-composition section) and drives Design decision #2 (liquidity
threshold). Selection-bias quantification (rubric: "was your sample
representative or biased?") also comes from this output.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

import config
from src import early_price

log = logging.getLogger(__name__)


def census_metadata(markets: pd.DataFrame) -> pd.DataFrame:
    """Counts by category x month from ingest metadata alone (no extra API calls)."""
    df = markets.copy()
    df["month"] = (pd.to_datetime(df["close_time"], utc=True)
                   .dt.tz_localize(None).dt.to_period("M").astype(str))
    df["resolved_yes"] = (df["result"] == "yes").astype(int)
    tab = (df.groupby(["category", "month"])
             .agg(n_markets=("ticker", "count"),
                  n_series=("series_ticker", "nunique"),
                  median_volume=("volume", "median"),
                  pct_resolved_yes=("resolved_yes", "mean"))
             .reset_index())
    return tab


def census_liquidity_sample(markets: pd.DataFrame) -> pd.DataFrame:
    """Pull opening candlesticks for a random subsample per category.

    Costs ~ CATEGORIES x SAMPLE x 1-2 requests; keeps M1 cheap while telling
    us whether the liquidity filter threshold is sane per category.
    """
    rng = random.Random(config.CENSUS_RANDOM_SEED)
    frames = []
    for cat, group in markets.groupby("category"):
        tickers = group["ticker"].tolist()
        rng.shuffle(tickers)
        take = tickers[: config.CENSUS_CANDLE_SAMPLE_PER_CATEGORY]
        feats, n_fail, first_err = [], 0, None
        by_ticker = group.set_index("ticker", drop=False)
        for i, t in enumerate(take):
            m = by_ticker.loc[t].to_dict()
            try:
                feats.append(early_price.early_price_features(m))
            except Exception as e:  # noqa: BLE001 - census must not die mid-pull
                n_fail += 1
                first_err = first_err or f"{t}: {type(e).__name__}: {e}"
                log.warning("candles failed for %s: %s", t, e)
            if (i + 1) % 10 == 0:
                log.info("[%s] sampled %d/%d", cat, i + 1, len(take))
        if n_fail == len(take):
            log.error("[%s] ALL %d candle pulls failed; first error: %s",
                      cat, n_fail, first_err)
            continue
        f = pd.DataFrame(feats)
        f["category"] = cat
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_liquidity(sample: pd.DataFrame) -> pd.DataFrame:
    def pct(s):
        return 100.0 * s.mean()
    return (sample.groupby("category")
            .agg(n_sampled=("ticker", "count"),
                 pct_has_first_trade=("first_trade_price",
                                      lambda s: pct(s.notna())),
                 pct_trade_within_10pct_life=("first_trade_frac",
                                              lambda s: pct(s.fillna(9) <= 0.10)),
                 pct_trade_within_25pct_life=("first_trade_frac",
                                              lambda s: pct(s.fillna(9) <= 0.25)),
                 median_first_trade_frac=("first_trade_frac", "median"),
                 median_open_spread=("open_spread", "median"),
                 median_life_hours=("life_minutes",
                                    lambda s: s.median() / 60.0))
            .reset_index())


def run(markets_path: str | None = None) -> None:
    path = markets_path or f"{config.PROCESSED_DIR}/markets.parquet"
    markets = pd.read_parquet(path)
    out = Path(config.PROCESSED_DIR)

    meta = census_metadata(markets)
    meta.to_csv(out / "census_by_category_month.csv", index=False)

    print("\n=== Markets by category ===")
    print(markets.groupby("category")["ticker"].count().to_string())

    sample = census_liquidity_sample(markets)
    if sample.empty:
        print("\nERROR: liquidity sampling failed for every market — "
              "see 'candles failed' warnings in the log.")
        return
    sample.to_csv(out / "census_liquidity_sample.csv", index=False)
    summary = summarize_liquidity(sample)
    summary.to_csv(out / "census_liquidity_summary.csv", index=False)
    print("\n=== Early-liquidity summary (sampled) ===")
    print(summary.to_string(index=False))
    print(f"\nFull tables written to {out}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run()
