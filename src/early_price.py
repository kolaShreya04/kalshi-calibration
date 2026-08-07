"""
Early-price construction v2 — census-driven redesign (see config.py note).

The M1 census showed Kalshi books are nearly empty at the opening bell, so
v2 centers on the FIRST TRADED PRICE and window definitions that scale with
market lifetime. Request budget is 1-2 candlestick calls per market:

  short markets (life <= ~3.4 days): one minute-level request for the whole
      relevant span -> exact first trade + all windows.
  long markets: one hourly request to locate the first traded hour, then
      one minute-level request inside that hour for the exact first trade.

Outputs per market:
  first_trade_price / first_trade_minute / first_trade_frac (of lifetime)
  first_day_vwap / first_day_volume        (first 1440 min)
  life10_vwap  / life10_volume             (first 10% of lifetime)
  open_spread / open_mid                   (first quoted candle)
  life_minutes, n_candles
"""

from __future__ import annotations

import logging
from typing import Optional

import config
from src import api

log = logging.getLogger(__name__)

# life <= this many minutes fits in one 5000-candle minute-level request
_ONE_SHOT_MINUTES = 4900


def _num(block: dict | None, key: str) -> Optional[float]:
    """Read candle value across live ('close_dollars') / historical ('close')."""
    if not block:
        return None
    v = block.get(f"{key}_dollars", block.get(key))
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _vol(c: dict) -> float:
    v = c.get("volume_fp", c.get("volume", 0))
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _trade_price(c: dict) -> Optional[float]:
    return _num(c.get("price"), "mean") or _num(c.get("price"), "close")


def _vwap(candles: list[dict], end_ts: int) -> tuple[Optional[float], float]:
    """(VWAP, volume) over traded candles with end_period_ts <= end_ts."""
    num = den = 0.0
    for c in candles:
        if c["end_period_ts"] > end_ts:
            break
        v, p = _vol(c), _trade_price(c)
        if v > 0 and p is not None:
            num += p * v
            den += v
    return (num / den if den > 0 else None), den


def early_price_features(market: dict) -> dict:
    open_dt = api.iso_to_dt(market["open_time"])
    close_dt = api.iso_to_dt(market["close_time"])
    open_ts, close_ts = api.dt_to_ts(open_dt), api.dt_to_ts(close_dt)
    life_min = max((close_ts - open_ts) / 60.0, 1.0)
    historical = api.is_historical(api.iso_to_dt(
        market.get("settlement_ts") or market["close_time"]))
    ticker, series = market["ticker"], market["series_ticker"]

    feats: dict = {"ticker": ticker, "life_minutes": life_min,
                   "first_trade_price": None, "first_trade_minute": None,
                   "first_trade_frac": None, "open_spread": None,
                   "open_mid": None, "n_candles": 0}

    if life_min <= _ONE_SHOT_MINUTES:
        minute = api.candlesticks(ticker, series, open_ts, close_ts,
                                  historical, interval_min=1)
        search = minute
    else:
        end = min(close_ts, open_ts + config.EARLY_MAX_SEARCH_HOURS * 3600)
        hourly = api.candlesticks(ticker, series, open_ts, end,
                                  historical, interval_min=60)
        search = hourly
        minute = []
        first_hour = next((c for c in hourly if _vol(c) > 0), None)
        if first_hour is not None:
            h_end = first_hour["end_period_ts"]
            minute = api.candlesticks(ticker, series, h_end - 3600, h_end,
                                      historical, interval_min=1)

    feats["n_candles"] = len(search)

    # opening quote state (first candle with a two-sided quote)
    for c in search:
        bid, ask = _num(c.get("yes_bid"), "close"), _num(c.get("yes_ask"), "close")
        if bid is not None and ask is not None and ask > 0:
            feats["open_spread"] = ask - bid
            feats["open_mid"] = (ask + bid) / 2.0
            break

    # exact first trade (minute resolution when available)
    fine = minute if minute else search
    for c in fine:
        if _vol(c) > 0 and _trade_price(c) is not None:
            feats["first_trade_price"] = _trade_price(c)
            feats["first_trade_minute"] = max(
                (c["end_period_ts"] - open_ts) / 60.0, 0.0)
            feats["first_trade_frac"] = feats["first_trade_minute"] / life_min
            break

    # window VWAPs from the coarse series (hourly for long markets)
    day_end = open_ts + config.FIRST_DAY_MINUTES * 60
    feats["first_day_vwap"], feats["first_day_volume"] = _vwap(search, day_end)
    life10_end = open_ts + max(3600, int(config.LIFE_FRACTION * life_min * 60))
    feats["life10_vwap"], feats["life10_volume"] = _vwap(search, life10_end)
    return feats


def usable_mask(df):
    """Core-sample filter (documented in report's Data Cleaning section):
    a genuine early price exists (first trade within EARLY_FRACTION_MAX of
    lifetime) and the market saw real lifetime activity."""
    return (df["first_trade_frac"].notna()
            & (df["first_trade_frac"] <= config.EARLY_FRACTION_MAX)
            & (df["volume"].fillna(0) >= config.MIN_TOTAL_VOLUME))
