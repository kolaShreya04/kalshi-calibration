"""
Central configuration for the Kalshi calibration project (ISyE 6740, Group 010).

Every analysis choice that could plausibly change lives here, so that
robustness re-runs (per peer-review feedback on early-price windowing)
are one-line edits that regenerate everything downstream.
"""

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Conservative unauthenticated rate limit (requests per second).
# Raise after checking your tier at GET /account/api-limits with an API key.
REQUESTS_PER_SECOND = 8.0  # client backs off automatically on HTTP 429
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0  # exponential base for 429/5xx retries

# ---------------------------------------------------------------------------
# Sample definition  (Design decision #6 in PROJECT_PLAN.md)
# ---------------------------------------------------------------------------
# M0 verified minute candlesticks exist back to at least Oct 2024.
SAMPLE_START = datetime(2024, 11, 1, tzinfo=timezone.utc)
SAMPLE_END = datetime(2026, 6, 30, tzinfo=timezone.utc)

# Kalshi series categories to ingest (verified valid on 2026-07-12).
# Category names are exact strings returned by GET /series?category=...
CATEGORIES = [
    "Politics",
    "Sports",
    "Crypto",
    "Economics",
    "Financials",
    "Mentions",
]

# ---------------------------------------------------------------------------
# Market-level filters  (documented in the report's Data Cleaning section)
# ---------------------------------------------------------------------------
# Exclude multivariate parlay wrapper markets (combos of other markets).
EXCLUDE_MVE = True
# Only binary yes/no markets with a definitive outcome.
VALID_RESULTS = {"yes", "no"}
# Liquidity filter (Design decision #2): minimum contracts traded in the
# early window for a market to enter the core sample. Tune after census.
MIN_EARLY_VOLUME = 10
# Minimum total lifetime volume (drops zero-activity strikes).
MIN_TOTAL_VOLUME = 100

# ---------------------------------------------------------------------------
# Early-price definitions  (Design decision #1; peer-review robustness axis)
#
# REVISED after the M1 census (2026-07-13): Kalshi order books are nearly
# empty at open (median opening spread 0.90-0.98 outside crypto/financials;
# <25% of markets trade in the first hour; median time to first trade ranges
# from ~25 min in crypto to ~10 hours in sports). A fixed "first hour VWAP"
# would discard ~90% of markets. Primary definition is therefore the FIRST
# TRADED PRICE, with two robustness definitions:
#   first_day_vwap : VWAP over first 1440 minutes
#   life10_vwap    : VWAP over the first 10% of the market's lifetime
#                    (normalizes across 15-min crypto vs 6-month politics)
# ---------------------------------------------------------------------------
PRIMARY_EARLY_DEF = "first_trade"
PRIMARY_PRICE_COL = "first_trade_price"
FIRST_DAY_MINUTES = 1440
LIFE_FRACTION = 0.10
# A first trade counts as "early" only if it occurs within this fraction of
# the market's lifetime (core-sample filter; sensitivity-checked in report).
EARLY_FRACTION_MAX = 0.25
# Hourly-candle search horizon for locating the first trade (~208 days).
EARLY_MAX_SEARCH_HOURS = 5000

# ---------------------------------------------------------------------------
# Census settings (M1 deliverable)
# ---------------------------------------------------------------------------
# Number of markets per category to sample for early-liquidity stats.
CENSUS_CANDLE_SAMPLE_PER_CATEGORY = 50
CENSUS_RANDOM_SEED = 6740

# Early-price collection subsample (ingest found ~10M markets; pulling
# candles for all is infeasible and statistically unnecessary).
# Stratified random per category, seeded -> reproducible, documented in
# the report's Data Cleaning section as a design decision.
SUBSAMPLE_PER_CATEGORY = 20000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"          # cached API responses (never re-pull)
PROCESSED_DIR = f"{DATA_DIR}/processed"
FIGURES_DIR = "figures"
