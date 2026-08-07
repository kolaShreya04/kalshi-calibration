# Kalshi Opening-Price Calibration — ISyE 6740 Group 010

Are Kalshi prediction markets well calibrated at open? Category-level study
across Politics, Sports, Crypto, Economics, Financials, and Mentions.
See `PROJECT_PLAN.md` (shared separately) for milestones and rubric strategy.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No API key is required — all market data endpoints are public
(verified 2026-07-12). If you hit rate limits on the full ingest, lower
`REQUESTS_PER_SECOND` in `config.py`, or generate a key at
kalshi.com → account → API keys and we'll add signed auth to `src/api.py`.

## Run order

```bash
python scripts/probe.py           # 1. 30s sanity check of API assumptions
python scripts/run_m1.py --smoke  # 2. fast end-to-end smoke run
python scripts/run_m1.py          # 3. full ingest + census (long; restartable)
python src/collect_early.py       # 4. early prices for ALL markets (longest pull;
                                  #    restartable; split with --category)
python scripts/run_m2.py          # 5. calibration + favorite-longshot analysis
python scripts/run_m3.py          # 6. residual prediction model (temporal CV)
python scripts/run_m4.py          # 7. drift + tradeability after fees/spread
python -m pytest tests/ -q        # offline tests (no network needed)
```

Outputs land in `data/processed/`:

| File | What it is |
|---|---|
| `markets.parquet` | One row per resolved binary market (the master table) |
| `filter_log.json` | Count of dropped markets by reason → report's Data Cleaning section |
| `census_by_category_month.csv` | Sample composition → report table |
| `census_liquidity_summary.csv` | Early-liquidity per category → sets `MIN_EARLY_VOLUME` |

## Repo layout

```
config.py          # ALL analysis choices (windows, filters, sample dates)
src/api.py         # rate-limited client + live/historical cutoff router
src/ingest.py      # settled-market ingest across categories (M1)
src/early_price.py # early-price definitions: first_trade / first_hour / first_day
src/census.py      # data census (M1 deliverable)
scripts/           # entry points
data/              # gitignored; raw/ = cached API pulls, processed/ = tables
```

## Key API facts baked into the code (M0, verified 2026-07-12)

- Data is partitioned at a moving cutoff (`GET /historical/cutoff`,
  currently 2026-03-08; live window ≈ 3 months). Pre-cutoff markets must be
  read from `/historical/...` — live endpoints return them missing or with
  zeroed stats. `src/api.py` routes automatically and falls back across tiers.
- Minute candlesticks include yes_bid/yes_ask OHLC even in minutes with no
  trades → spread at open is observable. Field names differ between live
  (`close_dollars`) and historical (`close`) tiers; `early_price._num()` handles both.
- Multivariate parlay wrappers (`mve_collection_ticker`) are excluded — they
  are combos of other markets, not unit markets.
- Historical depth confirmed to at least Oct 2024 (PRES-2024-DJT has full
  minute candles). Sample window is set in `config.py`.

## Team workflow

- `main` branch protected; work on `feature/<workstream>` branches, PR + one review.
- Never commit `data/` (gitignored). Each member regenerates locally —
  everything is deterministic from `config.py` + cached raw pulls.
- Workstream ownership per proposal §5: data/ingest, calibration+FLB,
  residual model + drift. Census results shared before choosing thresholds.
```
