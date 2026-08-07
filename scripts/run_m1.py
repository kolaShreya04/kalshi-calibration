"""
M1 entry point: ingest + census in one command.

Usage:
  python scripts/run_m1.py            # full sample window
  python scripts/run_m1.py --smoke    # quick smoke run (few pages per tier)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ingest, census  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="limit paging for a fast end-to-end check")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    max_pages = 3 if args.smoke else None
    ingest.ingest_markets(max_pages_per_tier=max_pages)
    census.run()


if __name__ == "__main__":
    main()
