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
