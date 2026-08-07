"""
Full early-price collection for every market in markets.parquet (M2 prereq).

This is the expensive pull (~1-2 candlestick requests per market), so it is
RESUMABLE: each market's features are appended to a JSONL cache keyed by
ticker; re-running skips everything already fetched. Kill it and restart
freely; teammates can also split the work by category via --category.

Output: data/processed/early_prices.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import config
from src import early_price

log = logging.getLogger(__name__)

CACHE = Path(config.RAW_DIR) / "early_features.jsonl"


def _load_done() -> set[str]:
    if not CACHE.exists():
        return set()
    done = set()
    with CACHE.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["ticker"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def subsample(markets: pd.DataFrame,
              per_category: int | None = None) -> pd.DataFrame:
    """Seeded stratified subsample: up to N markets per category, sampled
    uniformly at random within category (uniform over the whole sample
    window in expectation). Reproducible across machines via the seed, so
    teammates splitting by category draw from the same subsample."""
    n = per_category or config.SUBSAMPLE_PER_CATEGORY
    if not n:
        return markets
    # Sample only among markets that ever traded (census: ~96% of crypto/
    # financial strike-ladder markets never trade; the analysis filters on
    # lifetime volume anyway, so sampling untraded strikes wastes API calls).
    markets = markets[markets["volume"].fillna(0) >= config.MIN_TOTAL_VOLUME]
    parts = [g.sample(n=min(n, len(g)),
                      random_state=config.CENSUS_RANDOM_SEED)
             for _, g in markets.groupby("category")]
    out = pd.concat(parts, ignore_index=True)
    log.info("subsample: %d -> %d markets (cap %d per category)",
             len(markets), len(out), n)
    return out


def collect(category: str | None = None,
            per_category: int | None = None) -> None:
    markets = pd.read_parquet(f"{config.PROCESSED_DIR}/markets.parquet")
    markets = subsample(markets, per_category)
    if category:
        markets = markets[markets["category"] == category]
    done = _load_done()
    todo = markets[~markets["ticker"].isin(done)]
    log.info("early-price collection: %d done, %d to go", len(done), len(todo))

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as f:
        for i, (_, m) in enumerate(todo.iterrows()):
            try:
                feats = early_price.early_price_features(m.to_dict())
            except Exception as e:  # noqa: BLE001 - keep the long pull alive
                log.warning("failed %s: %s", m["ticker"], e)
                feats = {"ticker": m["ticker"], "error": str(e)}
            f.write(json.dumps(feats) + "\n")
            if (i + 1) % 100 == 0:
                f.flush()
                log.info("collected %d/%d", i + 1, len(todo))


def finalize() -> pd.DataFrame:
    """Merge JSONL cache into a parquet aligned with markets.parquet."""
    rows = []
    with CACHE.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    feats = pd.DataFrame(rows).drop_duplicates("ticker", keep="last")
    markets = pd.read_parquet(f"{config.PROCESSED_DIR}/markets.parquet")
    # inner join: early_prices.parquet contains only collected (subsampled)
    # markets — keeps the analysis file small and unambiguous
    df = markets.merge(feats, on="ticker", how="inner")
    df.to_parquet(f"{config.PROCESSED_DIR}/early_prices.parquet", index=False)
    n_priced = (df[config.PRIMARY_PRICE_COL].notna().sum()
                if config.PRIMARY_PRICE_COL in df.columns else 0)
    log.info("early_prices.parquet: %d rows, %d with a first-trade price",
             len(df), n_priced)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="restrict to one category (for splitting work)")
    parser.add_argument("--per-category", type=int, default=None,
                        help="override config.SUBSAMPLE_PER_CATEGORY (0 = no cap)")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not args.finalize_only:
        collect(args.category, args.per_category)
    finalize()
