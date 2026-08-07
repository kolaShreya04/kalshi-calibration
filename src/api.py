"""
Kalshi API client with rate limiting, retries, cursor pagination, and the
live/historical router discovered in M0.

Key facts this module encodes (verified 2026-07-12, see PROJECT_PLAN.md §7):
- Kalshi partitions data at a moving cutoff (GET /historical/cutoff).
  Markets settled before the cutoff are missing or stat-wiped on live
  endpoints and must come from /historical/... equivalents.
- Live candlesticks:  GET /series/{series}/markets/{ticker}/candlesticks
  Historical:         GET /historical/markets/{ticker}/candlesticks
- All public market data works unauthenticated.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})
_last_request_time = 0.0


def _throttle() -> None:
    """Simple token-spacing rate limiter."""
    global _last_request_time
    min_gap = 1.0 / config.REQUESTS_PER_SECOND
    now = time.monotonic()
    wait = _last_request_time + min_gap - now
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def get(path: str, params: dict | None = None) -> dict:
    """GET with throttling and exponential-backoff retries on 429/5xx AND
    network errors (timeouts, connection resets, DNS blips).

    Long unattended pulls die without this: a single 30s server stall used
    to raise ReadTimeout straight through and kill the whole pipeline.
    """
    url = f"{config.BASE_URL}/{path.lstrip('/')}"
    for attempt in range(config.MAX_RETRIES):
        sleep_s = config.RETRY_BACKOFF_SECONDS * (2 ** attempt)
        _throttle()
        try:
            resp = _session.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            log.warning("network error on %s (%s); retry in %.1fs",
                        path, type(e).__name__, sleep_s)
            time.sleep(sleep_s)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise KeyError(f"404 for {url} (params={params})")
        if resp.status_code == 429 or resp.status_code >= 500:
            log.warning("HTTP %s on %s; retry in %.1fs", resp.status_code, path, sleep_s)
            time.sleep(sleep_s)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Exhausted retries for {url}")


def paginate(path: str, params: dict | None = None, list_key: str = "markets",
             max_pages: int | None = None):
    """Yield items across cursor-paginated responses."""
    params = dict(params or {})
    pages = 0
    while True:
        data = get(path, params)
        items = data.get(list_key) or []
        yield from items
        cursor = data.get("cursor")
        pages += 1
        if not cursor or not items or (max_pages and pages >= max_pages):
            return
        params["cursor"] = cursor


# ---------------------------------------------------------------------------
# Cutoff router
# ---------------------------------------------------------------------------

_cutoff_cache: datetime | None = None


def market_settled_cutoff() -> datetime:
    """The boundary: markets settled before this need /historical endpoints."""
    global _cutoff_cache
    if _cutoff_cache is None:
        data = get("historical/cutoff")
        _cutoff_cache = datetime.fromisoformat(
            data["market_settled_ts"].replace("Z", "+00:00"))
        log.info("Historical cutoff: %s", _cutoff_cache)
    return _cutoff_cache


def is_historical(settled_or_close_time: datetime) -> bool:
    return settled_or_close_time < market_settled_cutoff()


# ---------------------------------------------------------------------------
# Endpoint wrappers
# ---------------------------------------------------------------------------

def list_series(category: str) -> list[dict]:
    """All series for a category (includes fee_type/fee_multiplier)."""
    data = get("series", {"category": category})
    return data.get("series") or []


def live_settled_markets(max_close_ts: int | None = None, **extra):
    params = {"status": "settled", "limit": 1000}
    if max_close_ts:
        params["max_close_ts"] = max_close_ts
    params.update(extra)
    return paginate("markets", params, "markets")


def historical_markets(**extra):
    params = {"limit": 1000}
    params.update(extra)
    return paginate("historical/markets", params, "markets")


def get_market(ticker: str, historical: bool) -> dict:
    path = f"historical/markets/{ticker}" if historical else f"markets/{ticker}"
    return get(path)["market"]


def candlesticks(ticker: str, series_ticker: str, start_ts: int, end_ts: int,
                 historical: bool, interval_min: int | None = None) -> list[dict]:
    """Minute candlesticks (price/yes_bid/yes_ask OHLC, volume, OI).

    Automatically falls back live->historical (and vice versa) because the
    archive boundary can lag the cutoff timestamp.
    """
    interval = interval_min or config.CANDLE_INTERVAL_MIN
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": interval}
    primary = (f"historical/markets/{ticker}/candlesticks" if historical
               else f"series/{series_ticker}/markets/{ticker}/candlesticks")
    fallback = (f"series/{series_ticker}/markets/{ticker}/candlesticks" if historical
                else f"historical/markets/{ticker}/candlesticks")
    for path in (primary, fallback):
        try:
            candles = get(path, params).get("candlesticks") or []
        except KeyError:
            candles = []
        if candles:
            return candles
    return []


# ---------------------------------------------------------------------------
# Raw-response caching (never re-pull; ingest is restartable)
# ---------------------------------------------------------------------------

def cache_json(relpath: str, payload) -> None:
    p = Path(config.RAW_DIR) / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))


def load_cached(relpath: str):
    p = Path(config.RAW_DIR) / relpath
    if p.exists():
        return json.loads(p.read_text())
    return None


def iso_to_dt(s) -> datetime:
    """Parse a timestamp that may be an ISO string OR an already-parsed
    datetime/pandas.Timestamp (parquet round-trips produce the latter)."""
    if s is None or s != s:  # None or NaN/NaT
        raise ValueError("missing timestamp")
    if isinstance(s, datetime):
        return s if s.tzinfo is not None else s.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def dt_to_ts(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp())
