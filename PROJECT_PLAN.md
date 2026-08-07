# Kalshi Calibration Project — Strategic Plan
**Group 010 · Shreya Kola, Rahi Patel, Kunj Shah · ISyE 6740**

---

## 1. Where we stand

The proposal is **done and scored 5/5 by every peer reviewer** (2/2 problem statement, 1/1 methodology, 1/1 evaluation, 1/1 formatting/length across all reviews). There is nothing to resubmit. All peer feedback gets incorporated into the **final report and implementation**, not a revised proposal.

The final report is graded 0–20 by peer review (TA-moderated), 5 points each:

| Rubric dimension | What earns 5/5 |
|---|---|
| Creativity & Scope | Novel angle + complexity matching a **3-person** group |
| Formulation | Grounded in existing theory/models, citations for anything outside course content |
| Rigor of Implementation | Technical quality, adaptation to challenges, brought to a **complete** state |
| Report Writing Quality | Logical flow, captioned figures/tables, clean formatting, no typos |

Constraints: ≥4 pages (excluding title page, references, large graphics), font 11, single-spaced, single column. Required sections: Problem Statement, Methodology, Evaluation/Results, plus a Data Cleaning/Extraction section if significant (ours is). The two student samples run ~10 pages — target **8–10 pages**, not 4.

## 2. Peer feedback → concrete actions

| Feedback (reviewer, date) | How we incorporate it |
|---|---|
| Early-window average may smooth out the microstructure inefficiencies we want to detect (Jul 1) | **Robustness analysis**: run all core analyses under 3 early-price definitions — first traded price, first-hour VWAP/mid, first-day VWAP/mid. Report sensitivity in a dedicated subsection. This is our single richest rigor win. |
| Naive baseline under-defined (Jul 1) | Define explicitly in the report: (a) the early price itself as the probability forecast (market baseline), and (b) a no-feature model predicting the category base rate. Residual model must beat (b) on AUC/log loss to claim systematic mispricing. |
| Specify which predictive models (Jul 1) | Name them up front: regularized logistic regression (interpretable anchor), random forest + gradient boosting (nonlinear), compared against the naive baseline. Ties directly to course content. |
| Consider intermediate prices (Jul 1) | Cheap extension of the drift workstream: sample price at open / mid-life / close, show how calibration error decays over a market's life. One figure, high creativity value. |
| Thin, irregular liquidity at the opening bell across categories (Jul 3) | Explicit liquidity filter (min trades/volume in early window), documented; sensitivity check on the threshold. Report how many markets each filter drops per category. |
| No extra points for data cleaning — limit extraction scope (Jul 3) | Build the minimum pipeline that supports the four workstreams. One clean pull → cached parquet. No heroics. Still include a short Data Cleaning section (rubric asks for it) but keep it tight. |
| "Explain your dimensionality reduction / explained variance plots" (Jun 30) | Almost certainly a misdirected review (our proposal has no dimensionality reduction). Ignore, but it's a reminder: peer graders skim — make every figure self-explanatory. |

## 3. Milestones

**M0 — De-risk the data (do first, ~days).** Verify via Kalshi's historical API that we can actually get: early-window trade/quote prices, bid-ask spread near open, volume/open interest, category metadata, settlement outcomes — for enough resolved markets in politics, sports, crypto/econ. *This is the project's only existential risk.* If early-window granularity is unavailable, we adapt the "early price" definition now, not in week 4.

**M1 — Data pipeline.** One script: pull resolved markets → clean → filter → cached parquet + a data dictionary. Deliverable: dataset summary table (markets per category, date range, liquidity stats) that goes straight into the report.

**M2 — Calibration + favorite-longshot (Workstreams 1–2).** Calibration curves, Brier, ECE, bootstrap CIs on bin-level resolution rates; signed deviation across probability range; all × category and × early-price definition.

**M3 — Residual prediction model (Workstream 3, the ML anchor).** Features: category, early volume, open interest, time-to-resolution, spread, price level. Price-level control as proposed. Temporal CV (no leakage), AUC + log loss vs. naive baseline, feature importances.

**M4 — Drift + tradeability (Workstream 4).** Open→close conditional drift, intermediate-price decay figure, paper-trading rule, gross vs. net edge after Kalshi fees + observed spread. A null result is a finding — frame it as evidence of efficiency, exactly as the proposal promised.

**M5 — Report + polish.** Assemble in the LaTeX template, internal peer-review pass against the rubric, proofread, export PDF.

Suggested split (mirrors proposal's Team Responsibilities): one member leads M1, one leads M2, one leads M3+M4; category breakdown and report shared. Report must state the partition — both samples do, and the rubric scales expectations by group size.

## 4. Architecture

```
kalshi-calibration/
├── config.py            # windows, filters, fee schedule, category map
├── data/                # cached parquet (gitignored) + data dictionary
├── src/
│   ├── ingest.py        # API pull → raw
│   ├── clean.py         # filters, early-price construction (all 3 defs)
│   ├── calibration.py   # curves, Brier, ECE, bootstrap
│   ├── flb.py           # favorite-longshot tests
│   ├── residual_model.py# features, temporal CV, models
│   ├── drift.py         # drift, fees, net edge
│   └── viz.py           # one styling function → consistent figures
├── notebooks/           # exploration only, nothing load-bearing
├── figures/             # numbered outputs the report imports
└── report/              # LaTeX (course template)
```

Principles: config-driven (window/filter changes = one line, reruns everything — this makes the robustness analysis nearly free), every report figure regenerable from a script, cache raw API responses so we never re-pull.

## 5. Decisions to lock before coding

1. **Early price**: first-hour mid as primary; first-trade and first-day as robustness. (Directly answers the strongest peer critique.)
2. **Liquidity filter**: minimum early-window trade count/volume — pick threshold after seeing M0 data.
3. **Unit of analysis**: one resolved binary market; define handling of multi-outcome event series (each leg separate vs. one leg per event) to avoid correlated-outcome double counting → at minimum, cluster bootstrap by event.
4. **Baselines**: market price + base-rate model (per feedback).
5. **Fee model**: Kalshi's published fee schedule, cited.
6. **Sample window**: fixed date range, stated in report.

## 6. Grade-maximization checklist

**Where points get lost (and how we don't):**
- *Rigor*: single train/test split, no uncertainty quantification (Fake-Job sample's weakness). We use temporal CV + bootstrap CIs throughout.
- *Rigor*: correlated outcomes — thousands of Kalshi markets aren't independent (same election, same game). Addressing this explicitly (cluster bootstrap) is the kind of thing a sophisticated peer reviewer probes; the rubric explicitly asks "was your sample representative or biased?"
- *Completeness*: rubric rewards "brought to its final complete state." All four workstreams must land, even if some results are null. Cut depth, never workstreams.
- *Writing*: peer graders are students who skim. Numbered, captioned, self-explanatory figures; no screenshot tables (Fake-Job sample); consistent styling via one viz function.
- *Formulation*: cite prior work — calibration literature (e.g., Murphy/Brier scoring), favorite-longshot bias literature, prediction-market efficiency papers. Proper citations, not bare URLs (NBA sample's weakness). Anything not covered in course content **requires** citations per the project guidelines.

**Commonly overlooked sections we will include:** partition of roles, data cleaning section, explicit Limitations subsection (NBA sample omitted one), references, appendix (extra figures + code link/repo).

**Presenting limitations professionally** (rubric-safe framing):
- Selection bias toward liquid/political markets → quantify it (sample composition table), state the population our claims cover.
- Historical edge ≠ live tradeability → already in the proposal's evaluation section; keep the "gap between measured and capturable edge" paragraph.
- Early-window noise → the three-definition robustness analysis converts this from a weakness into a rigor showcase.
- Null tradeability result → pre-framed in proposal §3.3 as a legitimate efficiency finding. Reiterate in results.
- Each limitation gets: acknowledgment → why it doesn't invalidate the core claim → concrete future-work item. Never list a limitation without a mitigation or a scoped claim.

## 7. M0 results (completed 2026-07-12) — data risk RETIRED

Verified live against Kalshi's public API (no API key required for any of this):

| Need | Endpoint | Status |
|---|---|---|
| Settled outcomes, open/close times, volume, OI | `GET /markets?status=settled` (live) | ✅ Confirmed |
| Category per market | `GET /series/{ticker}` → `category` field (+ `fee_type`!) | ✅ Confirmed ("Sports" etc.) |
| Early-window prices + **bid/ask** | `GET /series/{s}/markets/{t}/candlesticks?period_interval=1` — minute OHLC of price, yes_bid, yes_ask, volume, OI from the moment of open | ✅ Confirmed |
| Tick-level trades | `GET /markets/trades?ticker=…` (ticker required; global query returns empty) | ✅ Confirmed |
| Deep history | **Historical tier**: `GET /historical/markets`, `/historical/markets/{t}/candlesticks`, `/historical/trades` | ✅ Confirmed — PRES-2024-DJT (Oct 2024) returns full minute candlesticks with bid/ask |

**Critical architecture fact:** Kalshi partitions data at a moving cutoff (`GET /historical/cutoff`, currently **2026-03-08**; live window targets ~3 months). Markets settled before the cutoff are *invisible or stat-wiped* on live endpoints and must be fetched from `/historical/...`. The ingest layer needs a router: settled_ts < cutoff → historical endpoints, else live. Historical depth confirmed to **at least Oct 2024** → ~21 months of usable data.

Additional ingest rules discovered:
- **Filter out multivariate parlay wrappers** (`mve_collection_ticker` present, series `KXMVESPORTSMULTIGAME…`) — they pollute settled-market listings and are combos, not unit markets.
- Pre-cutoff market *metadata* from live `/markets` shows zeroed volume/prices (archived stubs, `updated_time` 2026-02-19) — always take pre-cutoff metadata from `/historical/markets`.
- `min_close_ts`+`max_close_ts` combined on live `/markets` behaved unreliably; prefer `max_close_ts` + cursor pagination.
- Fee model per series: `fee_type` (e.g. `quadratic_with_maker_fees`) and `fee_multiplier` come free from `/series` — cite Kalshi fee schedule for the formula.
- Rate limits are tiered (docs: `getting_started/rate_limits`); there's a **batch candlesticks endpoint** to cut request count. API key only needed if unauthenticated limits prove too slow — generate one anyway.

## 8. First steps (this week)

1. ~~M0 API spike~~ ✅ Done — see §7. All required data confirmed available, ~21 months deep.
2. Lock the six design decisions above as a team (sample window can now be generous: e.g. Nov 2024–Jun 2026).
3. Scaffold the repo + config, including the live/historical endpoint router.
4. M1 ingest against a small sample (one category, one month) before scaling; first output = data census table (markets/category/month, % with early liquidity).
