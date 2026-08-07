"""
30-second sanity probe: run this FIRST on your machine.

Re-verifies every M0 assumption the pipeline depends on, so if Kalshi
changes something we find out in 30 seconds, not after an hour of ingest.

Usage:  python scripts/probe.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import api  # noqa: E402


def check(name: str, fn):
    try:
        result = fn()
        print(f"  OK   {name}: {result}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {name}: {e}")
        return False


def main() -> None:
    print("Kalshi API probe\n" + "=" * 40)
    ok = True

    ok &= check("exchange status", lambda: api.get("exchange/status")["exchange_active"])
    ok &= check("historical cutoff", api.market_settled_cutoff)

    for cat in config.CATEGORIES:
        ok &= check(f"series category '{cat}'",
                    lambda c=cat: f"{len(api.list_series(c))} series")

    def one_settled():
        m = next(iter(api.live_settled_markets()))
        return m["ticker"]
    ok &= check("live settled markets", one_settled)

    def one_historical():
        m = next(iter(api.historical_markets()))
        return m["ticker"]
    ok &= check("historical markets", one_historical)

    def hist_candles():
        # PRES-2024-DJT opened 2024-10-04T12:15:00Z (M0 reference market)
        c = api.candlesticks("PRES-2024-DJT", "PRES",
                             1728044100, 1728047700, historical=True)
        return f"{len(c)} candles (expect ~17)"
    ok &= check("historical 1-min candlesticks w/ bid+ask", hist_candles)

    print("=" * 40)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - fix before ingest")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
