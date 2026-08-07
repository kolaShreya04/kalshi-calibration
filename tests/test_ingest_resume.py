"""
Ingest v2 checkpoint/resume validation with a simulated mid-walk crash.
Run: python tests/test_ingest_resume.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.chdir(tempfile.mkdtemp())  # isolate data/ writes

import pandas as pd

import config
from src import api, ingest


def make_page(page_idx: int, per_page: int = 100):
    """Fake API page of settled sports markets closing in 2026."""
    day = 28 - page_idx  # newest-first ordering
    return {
        "markets": [
            {"ticker": f"KXMLBGAME-26MAY{day:02d}TEST{i}-T{i}",
             "event_ticker": f"KXMLBGAME-26MAY{day:02d}TEST{i}",
             "market_type": "binary",
             "result": "yes" if i % 2 else "no",
             "open_time": f"2026-05-{day:02d}T10:00:00Z",
             "close_time": f"2026-05-{day:02d}T20:00:00Z",
             "settlement_ts": f"2026-05-{day:02d}T21:00:00Z",
             "volume_fp": "100.00", "open_interest_fp": "50.00"}
            for i in range(per_page)
        ],
        "cursor": f"CURSOR_{page_idx + 1}" if page_idx < 9 else "",
    }


class FakeAPI:
    """Serves 10 pages; crashes on a chosen page the first time."""
    def __init__(self, crash_on_page=None):
        self.crash_on_page = crash_on_page
        self.crashed = False

    def get(self, path, params=None):
        params = params or {}
        cur = params.get("cursor")
        page = int(cur.split("_")[1]) if cur else 0
        if page == self.crash_on_page and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated crash")
        return make_page(page)


def run():
    # keep test hermetic; checkpoint every 2 pages so the crash at page 6
    # lands after several checkpoints
    ingest.STATE_PATH = Path(config.RAW_DIR) / "ingest_state.json"
    ingest.CHECKPOINT_PAGES = 2
    series_map = pd.DataFrame([{"series_ticker": "KXMLBGAME",
                                "category": "Sports",
                                "fee_type": "quadratic",
                                "fee_multiplier": 0.07}])
    known = {"KXMLBGAME"}

    fake = FakeAPI(crash_on_page=6)
    orig_get = api.get
    api.get = fake.get
    try:
        state = ingest._load_state()
        try:
            ingest._walk_tier("live", "markets", {"limit": 100}, "markets",
                              known, state)
            raise AssertionError("should have crashed on page 6")
        except RuntimeError as e:
            assert "simulated crash" in str(e)

        # resume: fresh state load, walk again (fake no longer crashes)
        state2 = ingest._load_state()
        assert state2["tiers"]["live"]["cursor"] is not None
        scanned_before = state2["tiers"]["live"]["scanned"]
        assert scanned_before > 0
        ingest._walk_tier("live", "markets", {"limit": 100}, "markets",
                          known, state2)
        df = ingest.finalize(series_map)
    finally:
        api.get = orig_get

    # 10 pages x 100 rows, all unique, none double-ingested after resume
    assert len(df) == 1000, f"expected 1000 unique markets, got {len(df)}"
    assert df["ticker"].is_unique
    assert (df["category"] == "Sports").all()
    state3 = json.loads(ingest.STATE_PATH.read_text())
    assert state3["tiers"]["live"]["done"]
    print("PASS crash -> resume -> finalize (1000 unique rows, no dupes)")


if __name__ == "__main__":
    run()
    print("\n1 test passed")
