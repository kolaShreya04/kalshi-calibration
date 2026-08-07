"""
Offline tests using REAL API responses captured during M0 (2026-07-12).
Run: python -m pytest tests/ -q   (or python tests/test_offline.py)

These test parsing and filter logic without network access, so failures
mean code bugs, not API issues (probe.py covers the latter).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from collections import Counter
from src import early_price, ingest

# --- Real fixture: historical-tier candles (PRES-2024-DJT, first ~10 min) ---
HIST_CANDLES = [
    {"end_period_ts": 1728044160, "open_interest": "0.00", "volume": "0.00",
     "price": {"close": None, "mean": None},
     "yes_ask": {"close": "0.5000", "open": "1.0000"},
     "yes_bid": {"close": "0.4900", "open": "0.4900"}},
    {"end_period_ts": 1728044220, "open_interest": "1.00", "volume": "1.00",
     "price": {"close": "0.4900", "mean": "0.4900"},
     "yes_ask": {"close": "0.5100", "open": "0.5000"},
     "yes_bid": {"close": "0.4900", "open": "0.4900"}},
    {"end_period_ts": 1728044700, "open_interest": "100.00", "volume": "100.00",
     "price": {"close": "0.5100", "mean": "0.5100"},
     "yes_ask": {"close": "0.5100", "open": "0.5100"},
     "yes_bid": {"close": "0.4900", "open": "0.4900"}},
]

# --- Real fixture: live-tier candle (KXMLBGAME 2026, *_dollars format) ---
LIVE_CANDLE = {
    "end_period_ts": 1779654660, "open_interest_fp": "1.00", "volume_fp": "1.00",
    "price": {"close_dollars": "0.4900", "mean_dollars": "0.4900"},
    "yes_ask": {"close_dollars": "0.7300", "open_dollars": "0.4900"},
    "yes_bid": {"close_dollars": "0.1700", "open_dollars": "0.4100"},
}


def test_num_handles_both_tiers():
    assert early_price._num(HIST_CANDLES[1]["price"], "close") == 0.49
    assert early_price._num(LIVE_CANDLE["price"], "close") == 0.49
    assert early_price._num(HIST_CANDLES[0]["price"], "close") is None
    assert early_price._num(None, "close") is None


def test_candle_volume_both_tiers():
    assert early_price._vol(HIST_CANDLES[2]) == 100.0
    assert early_price._vol(LIVE_CANDLE) == 1.0
    assert early_price._vol({"volume": None}) == 0.0


def test_early_features_from_hist_fixture():
    # short market (2880 min life) -> one-shot minute branch
    market = {"ticker": "PRES-2024-DJT", "series_ticker": "PRES",
              "open_time": "2024-10-04T12:15:00Z",
              "close_time": "2024-10-06T12:15:00Z",
              "settlement_ts": "2024-10-06T13:00:00Z"}
    # bypass network: patch candlesticks + cutoff router
    from src import api
    orig_c, orig_h = api.candlesticks, api.is_historical
    api.candlesticks = lambda *a, **k: HIST_CANDLES
    api.is_historical = lambda dt: True
    try:
        f = early_price.early_price_features(market)
    finally:
        api.candlesticks, api.is_historical = orig_c, orig_h
    assert f["first_trade_price"] == 0.49
    assert abs(f["first_trade_minute"] - 2.0) < 0.01   # 12:17 end vs 12:15 open
    assert abs(f["first_trade_frac"] - 2.0 / 2880) < 1e-6
    assert f["open_spread"] is not None and abs(f["open_spread"] - 0.01) < 1e-9
    expected_vwap = (0.49 * 1 + 0.51 * 100) / 101
    assert abs(f["first_day_vwap"] - expected_vwap) < 1e-9
    assert f["first_day_volume"] == 101.0
    assert abs(f["life10_vwap"] - expected_vwap) < 1e-9
    assert f["life_minutes"] == 2880.0


# --- Real fixture: market objects from M0 ---
MVE_MARKET = {"ticker": "KXMVE-XYZ-ABC", "market_type": "binary", "result": "no",
              "close_time": "2026-05-27T22:00:00Z",
              "mve_collection_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-R"}
GOOD_MARKET = {"ticker": "KXMLBGAME-26MAY271610PHISD-PHI", "market_type": "binary",
               "result": "yes", "close_time": "2026-05-27T22:39:59Z",
               "event_ticker": "KXMLBGAME-26MAY271610PHISD",
               "volume_fp": "3309604.78", "open_interest_fp": "2079416.87",
               "open_time": "2024-05-24T20:26:00Z"}
VOID_MARKET = {"ticker": "X-1", "market_type": "binary", "result": "",
               "close_time": "2026-05-27T22:00:00Z"}
OLD_MARKET = {"ticker": "OLD-1", "market_type": "binary", "result": "no",
              "close_time": "2019-01-01T00:00:00Z"}


def test_ingest_filters():
    drop = Counter()
    assert not ingest._passes_filters(MVE_MARKET, drop)
    assert ingest._passes_filters(GOOD_MARKET, drop)
    assert not ingest._passes_filters(VOID_MARKET, drop)
    assert not ingest._passes_filters(OLD_MARKET, drop)
    assert drop["mve_parlay_wrapper"] == 1
    assert drop["no_yes_no_result"] == 1
    assert drop["outside_sample_window"] == 1


def test_series_prefix():
    assert ingest._series_prefix("KXMLBGAME-26MAY271610PHISD-PHI") == "KXMLBGAME"
    assert ingest._series_prefix("PRES-2024-DJT") == "PRES"


def test_row_from_market_numeric_coercion():
    row = ingest._row_from_market(GOOD_MARKET, "live")
    assert row["volume"] == 3309604.78
    assert row["series_ticker"] == "KXMLBGAME"
    assert row["source_tier"] == "live"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
