# Decision log

A running, chronological log. Each entry: **decision**, **alternatives considered**, **why**.
Entries were written when the decision was made (git history shows the order; the problem
definition, D-08 to D-16, was committed *before* any model code existed).

---

## Phase 0: Environment

**D-01: Python 3.13 in a `.venv`, versions pinned from `pip freeze`.**
Alternatives: conda; Docker. Why: the machine had Python 3.13.18. Every required library (including
Prophet 1.5.0 and statsforecast 2.1.1) installed from wheels without fighting, so plain `venv` + pinned
`requirements.txt` is the lightest reproducible option. Docker would be more hermetic but is overkill for a
portfolio repo the owner must run on a laptop.

**D-02: Push target.**
The build environment reaches GitHub over HTTPS through a proxy (SSH-style remotes are rewritten to HTTPS).
The session is authorised to push to the branch `ccr-8738f54c-7tbpae` of `PiyushDas-2004/RetailPulse`, not
to `main` directly. All work is committed there, and the final message gives the exact commands to make it
`main`. We never force-push.

## Phase 1: Dataset

**D-03: Dataset = M5 Forecasting (Walmart), obtained via the Nixtla `datasetsforecast` public mirror.**
Candidates tried, in the order the brief required:

| # | Candidate | Tried? | Outcome |
|---|-----------|--------|---------|
| a | M5 via `datasetsforecast` (`https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip`) | yes | The raw-file URL returned HTTP 403 *from the build environment's proxy* (a sandbox network policy, not a problem with the file). The same file was fetched by a sparse, shallow `git clone` of the same public repo. SHA-256 `cc704ba1…eed774`, 50.2 MB zip, 8 CSVs. **Accepted.** |
| b | UCI Online Retail II | reachability only (HTTP 200 on the zip) | Not needed because (a) passed every criterion. It would also have been weaker: one UK online gift retailer, about 2 years, transaction-level data that must be heavily cleaned, and no price/calendar side tables. |
| c | Rossmann / Favorita (Kaggle) | no | `~/.kaggle/kaggle.json` does not exist, so the Kaggle CLI cannot authenticate. |
| d | other | no | not needed |

Selection criteria verdicts (numbers from `reports/data_profile.json`):
- *>= 2 full years*: 1,941 days, 2011-01-29 to 2016-05-22, which is 277 complete weeks (5.3 years). **Pass.**
- *Regular, gap-free axis*: 0 calendar gaps, 0 missing sales cells. **Pass.**
- *20+ series*: 30,490 item x store series across 10 stores, 3 states, 3 categories and 7 departments. **Pass.**
- *Real retail demand*: unit sales recorded at real Walmart stores. **Pass.**
- *Licence*: released by Walmart/University of Nicosia for the M5 competition; the M5 organisers'
  repository and the Nixtla mirror distribute it for research use (see README for citation). We do not
  redistribute the raw data, only a download script plus small derived aggregates. **Acceptable for a
  non-commercial portfolio.**

Honest caveat for the Trent context: M5 contains no apparel. Its categories are FOODS, HOBBIES and
HOUSEHOLD. The *methods* (hierarchical store/category forecasting, intermittent SKU demand, seasonality,
replenishment) transfer to fashion retail; the *numbers* do not.

**D-04: The download script tries HTTP first, then a sparse git clone, and verifies a SHA-256.**
Why: on a normal laptop the HTTP route works, and the git route is a robust fallback. The checksum
guarantees everyone models exactly the same bytes.

**D-05: Rebuild the `id` column.** The Nixtla mirror drops Kaggle's redundant `id` column
(`item_id + store_id + "_evaluation"`); we rebuild it losslessly.

**D-06: "Not listed" is treated as *no data*, not as zero demand.**
`sell_prices.csv` has a row only for weeks in which an item was on sale in a store. We verified that
**0 units** were sold in weeks with no price row, so the price table is a clean "listed" mask. Before an
item's first listed week, a zero means "the product did not exist in this store", and treating it as
"demand = 0" would bias every model toward zero. SKU series therefore start at their first listed week.
We found no item that was delisted and later relisted, so no internal gaps needed handling.

**D-07: Weekly buckets = Walmart fiscal weeks (Saturday to Friday). 277 complete weeks; the final 2 days are dropped.**
Why: a partial final week would look like a demand collapse. `calendar.wm_yr_wk` was asserted to be
constant within each 7-day block, and each block starts on a Saturday.

## Phase 2: Problem definition (written before any model code)

**D-08: Target = units sold.**
Alternatives: revenue. Why: inventory decisions (safety stock, reorder points) are made in units; revenue
mixes demand with price changes. Revenue is a units forecast times a price plan, so finance can derive it.

**D-09: Granularity = weekly.**
Alternatives: daily (the native grain), monthly. Why: daily item-store data is 68% zeros (profile) and
dominated by day-of-week patterns that replenishment does not need; stores order weekly. Weekly
aggregation cuts the zero share for listed item-store weeks to 23.9% and keeps 277 points, which is about
5.3 annual cycles. Monthly would leave only about 64 points and blur the Thanksgiving and Christmas weeks.
Cost: s = 52 is a long seasonal period, which is awkward for classical SARIMA and ETS (see D-20).

**D-10: Series set (211 series, 6 levels).**
1 total + 3 categories + 7 departments + 10 stores + 70 store x department (all 91 aggregate series), plus
**120 item x store (SKU) series**: 30 per Syntetos-Boylan class (smooth / erratic / intermittent / lumpy),
each split 10/10/10 over low/mid/high volume terciles within the class, sampled with seed 42.
Why stratify: a random sample would be about 51% smooth (10,360 of 20,366 eligible) and contain few
erratic SKUs. Equal classes give every demand type enough series for a per-class comparison, which is
the core of the "why it works here" analysis. Eligibility: listed for at least 200 of 277 weeks and no
gaps, so every SKU has at least 2 seasonal cycles before the first forecast origin. This excludes
new-product cold starts, which are out of scope (Limitations). SBC classes are computed on
**pre-holdout data only**, so the holdout never influences sampling.

**D-11: Horizon h = 13 weeks (one quarter).**
Alternatives: 4 weeks (M5 used 28 days), 8 weeks, 26 weeks. Why: 13 weeks matches a quarterly
merchandise-planning/buying cycle. The first 1 to 5 weeks of the same forecast feed weekly replenishment,
since the lead time plus review period is at most 5 weeks in our sensitivity grid.

**D-12: Protocol = rolling-origin (expanding-window) CV with 4 folds of 13 weeks, plus a final untouched 13-week holdout.**
Weeks are 0-indexed 0..276. Holdout = weeks 264-276 (2016-02-20 to 2016-05-14). CV test windows = 212-224,
225-237, 238-250, 251-263; each fold trains on all weeks before its window. Two extra *tuning* origins
(test windows 186-198 and 199-211) are used **only** to tune LightGBM hyper-parameters, so neither the CV
folds nor the holdout ever see tuning. A random split is invalid because training on the future to
predict the past leaks information (for example a lag-52 feature would contain the "test" value) and breaks
the autocorrelation structure. A single split is fragile: one quarter can be unusually easy or hard
(for example containing Christmas), and 4 folds + holdout span about 1 year of different seasons.

**D-13: Metrics.** WAPE = sum|e| / sum y is the primary metric: it is scale-free, defined when some
actuals are zero, and weights errors by volume, which is how a business feels them. Also reported:
MAE, RMSE, MASE (MAE scaled by the in-sample MAE of the seasonal-naive forecast, so it is comparable across
series and < 1 means "beats seasonal naive"), bias = sum(f - y) / sum y (positive means over-forecast,
which leads to overstock; negative means stock-outs), and sMAPE (caveat: asymmetric and unstable near 0).
Plain MAPE is not used because it divides by each actual, so it is undefined for the 23.9% zero SKU-weeks
and explodes for small actuals. Pooling: WAPE is computed per series, then summarised per level with the
**median across series**, so one huge series cannot dominate. A pooled WAPE per level is also reported.

**D-14: Baselines that must be beaten:** naive (last value), seasonal naive (same week last year, lag 52),
and a 4-week moving average. A model that cannot beat seasonal naive is not worth deploying, because
"same week last year" costs nothing to run, needs no maintenance, and is what many planners already use.

**D-15: Characteristics are computed on pre-holdout data** (weeks before 264): ADI, CV^2, share of zeros,
CV, STL trend and seasonal strength. Why: these are explanatory variables for the later analysis, and
they must not peek at the holdout.

**D-16: Seed = 42 everywhere** (numpy, LightGBM, sampling). Statistical models are deterministic given the data.

## Phase 3: EDA

**D-17: Leave STL "outliers" in place.** 38 pre-holdout weeks exceed |robust z| 3.5 on the STL remainder, but they come
in long runs (May-Sep 2012), which is a level/amplitude shift, not bad data. Capping would erase real behaviour; deleting
would create gaps. Models with a stochastic level adapt.

**D-18: Keep suspected stock-out zero runs.** 59/120 SKUs have zero runs of at least 4 weeks despite normally selling at
least 5 units per non-zero week. True demand is unknown without inventory records, so we model sales and state the
censoring limitation.

**D-19: No Box-Cox/log transform by default.** The total's yearly standard deviation stays at 16k-19k while the mean
rose 54%. This was confirmed out of sample for SARIMA on TOTAL (log 0.306 vs none 0.309 CV MASE: negligible).

## Phase 4: Models

**D-20: SARIMA engine = statsforecast AutoARIMA (s = 52, `approximation=True`).** statsmodels SARIMAX with s = 52 did
not finish a single fit in 10 minutes (large state vector); the exact AutoARIMA search takes 17-30 s per series, and
the CSS-approximated search 2-4 s. R's `auto.arima` uses the approximation by default for series this long or with
this period. The deep dive shows the top orders are within noise of each other, so the approximation costs little.

**D-21: Add dynamic harmonic regression (DHR-ARIMA).** It fixes the s = 52 vs 52.18-week mismatch and moving holidays,
and adds the SNAP monthly cycle found in EDA. K ∈ {2..10} is chosen once per series by AICc on pre-CV data, to
save compute (ARIMA orders are still re-selected at every origin).

**D-22: ETS via statsmodels, not statsforecast.** statsforecast's AutoETS silently drops seasonality for s > 24 (as R's
`ets` does), which was verified in its source. Seasonal states use heuristic initialisation; AICc counts the 51 free
seasonal states so that seasonal and non-seasonal candidates are compared fairly.

**D-23: LightGBM = one global, direct multi-horizon model** with the target scaled by the trailing 52-week mean.
Alternatives: one model per series (too little data per SKU), or a recursive strategy (error compounding; future
exogenous inputs would need forecasting). Training labels are winsorised at the 99.9th percentile because near-zero
trailing scales produced ratios above 20.

**D-24: LightGBM tuned only on two tuning origins before CV fold 1.** This keeps the CV and holdout comparisons fair
against statistical models, which choose their own orders in-sample by AICc.

**D-25: Ensemble = mean(best statistical model by CV, LightGBM).** The components were chosen from CV results only
(SARIMA ranked first among the statistical models, at 0.701).

**D-26: Clip all forecasts and interval bounds at 0, identically for every model.** Negative demand is impossible.

## Phase 5-6: Evaluation and forecasting

**D-27: Summaries use the median WAPE across series, pooled WAPE within groups, and mean MASE for ranking.** The
median and the MASE prevent one huge series from dominating; pooled WAPE gives the volume-weighted business view.

**D-28: Significance via a per-series DM test (HLN, lag h-1) plus a paired bootstrap with series as the resampling
unit.** Errors within a series are dependent, so series are the right unit to resample.

**D-29: Final model per series = the best mean CV MASE at its hierarchy level** (selection by level, not by series).
The why-analysis shows per-series winners are unstable (12.8% CV→holdout agreement), so selecting per series would
chase noise.

**D-30: Future SNAP days come from the fixed day-of-month rule** (verified on all 1,969 days). Future prices are the last
observed price carried forward (assumption).

## Phase 7: Inventory

**D-31: Periodic review (R = 1 week), order-up-to, lost sales.** Weekly ordering matches the forecast grain; lost sales
is realistic for retail shoppers. Lead time {1,2,3,4} weeks and service level {90,95,98}% are swept because the
data has no lead times. Costs: holding 1 per unit-week vs shortage 9 per unit (assumed).

**D-32: σ = RMSE of each model's own CV errors at h = 1-4.** Errors at these horizons match the protection interval
(≤ 5 weeks). A "demand-sigma" policy is included to show the cost of the textbook mistake.

**D-33: Weekly forecast updates without re-estimation in the simulation** (SARIMA `forward()`, LightGBM refreshed
features). This mirrors a weekly replenishment run that re-estimates models only periodically.

## Phase 8-11: Delivery

**D-34: `fact_backtest_forecasts.csv` (12 MB) is git-ignored and regenerated by `make export`;** all other outputs are
committed (≈ 20 MB in total). The Streamlit app degrades gracefully without it.

**D-35: REPORT.md tables are rendered from CSVs by `scripts/render_report.py`,** so no table is hand-typed.

**D-36: Work is pushed to the authorised session branch `ccr-8738f54c-7tbpae`** (the remote had no commits). Making it
`main` is one fast-forward push, documented in the final hand-off.
