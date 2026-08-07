"""
M1 ingest v2: settled-market table across categories — CHECKPOINTED.

v2 changes (after a ReadTimeout killed a 4.5h v1 run at 3.2M kept rows):
- RESUMABLE: cursor + counters checkpointed to data/raw/ingest_state.json
  every CHECKPOINT_PAGES pages; kept rows stream to per-tier JSONL files.
  Re-running continues from the last checkpoint instead of rescanning.
- LOW MEMORY: nothing accumulates in RAM; dedupe happens at finalize.

Strategy (see PROJECT_PLAN.md §7):
1. GET /series?category=X -> authoritative series -> category map + fees.
2. Walk BOTH listings globally (live /markets?status=settled and
   /historical/markets), newest-first, stopping once close_time passes
   SAMPLE_START. Per-series+timestamp filters were unreliable in M0
   probing, so we scan-and-filter locally by series prefix.
3. Filters: binary, yes/no result, non-MVE, in-window, category in scope.
4. finalize(): dedupe (live wins), join category map, write
   data/processed/markets.parquet + filter_log.json.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from src import api

log = logging.getLogger(__name__)

CHECKPOINT_PAGES = 20          # persist state every N pages (N*1000 markets)
STATE_PATH = Path(config.RAW_DIR) / "ingest_state.json"

MARKET_COLUMNS = [
    "ticker", "event_ticker", "series_ticker", "category",
    "result", "market_type", "strike_type",
    "open_time", "close_time", "settlement_ts",
    "volume", "open_interest", "liquidity",
    "fee_type", "fee_multiplier", "source_tier",
]


def build_series_map() -> pd.DataFrame:
    """Series ticker -> category + fee metadata, cached to raw/."""
    cached = api.load_cached("series_map.json")
    if cached is None:
        rows = []
        for cat in config.CATEGORIES:
            for s in api.list_series(cat):
                rows.append({
                    "series_ticker": s["ticker"],
                    "category": s.get("category", cat),
                    "fee_type": s.get("fee_type"),
                    "fee_multiplier": s.get("fee_multiplier"),
                })
            log.info("category %-12s: %d series", cat,
                     sum(r["category"] == cat for r in rows))
        api.cache_json("series_map.json", rows)
        cached = rows
    return pd.DataFrame(cached).drop_duplicates("series_ticker")


def _series_prefix(ticker: str) -> str:
    return ticker.split("-", 1)[0]


def _row_from_market(m: dict, source_tier: str) -> dict:
    def fp(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    return {
        "ticker": m.get("ticker"),
        "event_ticker": m.get("event_ticker"),
        "series_ticker": _series_prefix(m.get("ticker", "")),
        "result": m.get("result"),
        "market_type": m.get("market_type"),
        "strike_type": m.get("strike_type"),
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "settlement_ts": m.get("settlement_ts"),
        "volume": fp(m.get("volume_fp") or m.get("volume")),
        "open_interest": fp(m.get("open_interest_fp") or m.get("open_interest")),
        "liquidity": fp(m.get("liquidity_dollars")),
        "source_tier": source_tier,
    }


def _passes_filters(m: dict, drop: Counter) -> bool:
    if config.EXCLUDE_MVE and (m.get("mve_collection_ticker") or m.get("mve_selected_legs")):
        drop["mve_parlay_wrapper"] += 1
        return False
    if m.get("market_type") != "binary":
        drop["non_binary"] += 1
        return False
    if m.get("result") not in config.VALID_RESULTS:
        drop["no_yes_no_result"] += 1
        return False
    close = m.get("close_time")
    if not close:
        drop["missing_close_time"] += 1
        return False
    dt = api.iso_to_dt(close)
    if not (config.SAMPLE_START <= dt <= config.SAMPLE_END):
        drop["outside_sample_window"] += 1
        return False
    return True


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"tiers": {}, "drop": {}}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE_PATH)


def _tier_rows_path(tier: str) -> Path:
    return Path(config.RAW_DIR) / f"markets_{tier}.jsonl"


def _walk_tier(tier: str, path: str, base_params: dict, list_key: str,
               known_series: set, state: dict,
               max_pages: int | None = None) -> None:
    """Walk one listing tier with cursor checkpointing + row streaming."""
    tstate = state["tiers"].setdefault(
        tier, {"cursor": None, "scanned": 0, "kept": 0, "done": False})
    if tstate["done"]:
        log.info("[%s] already complete (%d kept), skipping",
                 tier, tstate["kept"])
        return
    drop = Counter(state.get("drop", {}))
    params = dict(base_params)
    if tstate["cursor"]:
        params["cursor"] = tstate["cursor"]
        log.info("[%s] resuming from checkpoint: %d scanned, %d kept",
                 tier, tstate["scanned"], tstate["kept"])

    rows_file = _tier_rows_path(tier)
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    pages_since_ckpt = 0
    pages_total = 0
    with rows_file.open("a") as out:
        while True:
            if max_pages and pages_total >= max_pages:
                tstate["done"] = True  # smoke runs treat partial as done
                state["drop"] = dict(drop)
                _save_state(state)
                log.info("[%s] stopped at max_pages=%d (smoke mode)",
                         tier, max_pages)
                return
            pages_total += 1
            data = api.get(path, params)
            items = data.get(list_key) or []
            stop = False
            for m in items:
                tstate["scanned"] += 1
                close = m.get("close_time")
                if close and api.iso_to_dt(close) < config.SAMPLE_START:
                    drop["older_than_window_stop"] += 1
                    stop = True
                    break
                if not _passes_filters(m, drop):
                    continue
                row = _row_from_market(m, tier)
                if row["series_ticker"] not in known_series:
                    drop["category_not_in_scope"] += 1
                    continue
                out.write(json.dumps(row) + "\n")
                tstate["kept"] += 1
            cursor = data.get("cursor")
            params["cursor"] = cursor
            pages_since_ckpt += 1
            if tstate["scanned"] % 100000 < 1000:
                log.info("[%s] scanned %d markets, kept %d",
                         tier, tstate["scanned"], tstate["kept"])
            if stop or not cursor or not items:
                tstate["done"] = True
            if pages_since_ckpt >= CHECKPOINT_PAGES or tstate["done"]:
                out.flush()
                tstate["cursor"] = cursor
                state["drop"] = dict(drop)
                _save_state(state)
                pages_since_ckpt = 0
            if tstate["done"]:
                log.info("[%s] complete: %d scanned, %d kept",
                         tier, tstate["scanned"], tstate["kept"])
                return


def finalize(series_map: pd.DataFrame) -> pd.DataFrame:
    """Merge tier JSONLs -> dedup -> join categories -> markets.parquet."""
    frames = []
    for tier in ("live", "historical"):
        p = _tier_rows_path(tier)
        if p.exists():
            frames.append(pd.read_json(p, lines=True))
    if not frames:
        raise RuntimeError("no ingested rows found")
    df = pd.concat(frames, ignore_index=True)
    # live wins on overlap: sort so live rows come first, keep first
    df["_tier_rank"] = (df["source_tier"] != "live").astype(int)
    df = (df.sort_values("_tier_rank").drop_duplicates("ticker")
            .drop(columns="_tier_rank"))
    df = df.merge(series_map, on="series_ticker", how="left")

    out = Path(config.PROCESSED_DIR)
    out.mkdir(parents=True, exist_ok=True)
    df = df[[c for c in MARKET_COLUMNS if c in df.columns]]
    df.to_parquet(out / "markets.parquet", index=False)
    state = _load_state()
    (out / "filter_log.json").write_text(json.dumps(state.get("drop", {}),
                                                    indent=2))
    log.info("Ingest complete: %d markets kept, drops=%s",
             len(df), state.get("drop", {}))
    return df


def ingest_markets(max_pages_per_tier: int | None = None) -> pd.DataFrame:
    series_map = build_series_map()
    known = set(series_map["series_ticker"])
    state = _load_state()

    _walk_tier("live", "markets",
               {"status": "settled", "limit": 1000,
                "max_close_ts": api.dt_to_ts(config.SAMPLE_END)},
               "markets", known, state, max_pages=max_pages_per_tier)
    _walk_tier("historical", "historical/markets", {"limit": 1000},
               "markets", known, state, max_pages=max_pages_per_tier)
    return finalize(series_map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ingest_markets()
