# RetailPulse: Retail Demand Forecasting & Inventory Optimization

*A reproducible study on public Walmart (M5) data. Author: Piyush Das. Every number in this report was produced
by the code in this repository. Tables are inserted automatically from `reports/tables/*.csv` by
`scripts/render_report.py`; the narrative numbers were checked against those tables.*

> **Read this first.** This is public US data (Walmart, 2011-2016, food/household/hobbies), **not** Trent,
> Westside or Zudio data. The inventory results come from a **simulation** under stated assumptions; they are not
> realised savings. Forecasts are probabilistic estimates, not guarantees.

---

## Contents
1. Executive summary
2. Business context
3. The dataset
4. Problem definition
5. Exploratory analysis
6. Stationarity and decomposition: implications for model choice
7. Models
8. SARIMA deep dive
9. Results
10. Why each model works where it does (and not elsewhere)
11. Final forecast
12. Inventory optimization
13. Limitations and threats to validity
14. Conclusions and next steps with real company data
15. Appendix: environment, reproduction, test results, glossary, full tables, interview Q&A

---

## 1. Executive summary

**Problem.** Forecast weekly unit demand 13 weeks (one quarter) ahead across a retail hierarchy, and turn the
forecasts into replenishment decisions (safety stock, reorder point, order-up-to level).

**Data.** M5 (Walmart): 30,490 item x store daily series, 2011-01-29 to 2016-05-22. These were aggregated into 277
complete Walmart weeks, and **211 series** were modelled: the total, 3 categories, 7 departments, 10 stores, 70
store x department series, and 120 item x store (SKU) series stratified 30 per demand class (smooth / erratic /
intermittent / lumpy).

**Protocol.** Ten models evaluated identically: 4 rolling-origin cross-validation folds (13-week windows) plus a
final untouched 13-week holdout. Metrics: WAPE (primary for volume-weighted error), MASE (cross-series ranking),
bias, RMSE, MAE, sMAPE, interval coverage. Differences are tested with Diebold-Mariano tests and a paired bootstrap.

**Headline findings (all from executed code):**
- **Best overall:** an equal-weight **ensemble of SARIMA and a global LightGBM**. CV mean MASE **0.681** (holdout
  0.668), against **1.020** (holdout 1.051) for seasonal naive and 0.860 for naive. It beats SARIMA on 68% of
  series (bootstrap ΔMASE −0.020, 95% CI [−0.030, −0.011]).
- **SARIMA is the best single model on the 91 aggregate series**: CV mean MASE 0.636, pooled WAPE 5.3% vs 10.2%
  for seasonal naive. It is significantly better than LightGBM there (ΔMASE −0.050, CI [−0.084, −0.017]) and
  than ETS (−0.089). It is clearly best at department level (0.519).
- **On the 120 SKU series a global LightGBM is the best single model** (0.733 vs SARIMA 0.751). The advantage is
  small (CI [−0.004, +0.040] includes 0) and survives removing the price features (0.741).
- **The owner's prior was confirmed with nuance.** SARIMA is strong on seasonal, trending, zero-free aggregates
  (it beats LightGBM on 66% of series with seasonal strength > 0.6, but on only 38% with seasonal strength ≤ 0.2).
  A Fourier/holiday/SNAP regression with ARIMA errors (DHR-ARIMA) is **statistically tied** with it (CI
  [−0.023, +0.010]). On noisy, zero-heavy SKUs SARIMA's advantage disappears.
- **Seasonal naive is the worst model here** (aggregate median bias −11.6%): five years of strong growth make
  "same week last year" systematically too low.
- **Per-series model choice is mostly noise**: the CV-best model is also holdout-best on only 12.8% of series;
  most per-series DM tests are not significant. The robust differences are at the group level.
- **Interval calibration varies a lot:** SARIMA, DHR-ARIMA and LightGBM are close to nominal (about 83-86% inside
  80% intervals), while ETS (50%) and Prophet (58%) are badly too narrow on aggregates.
- **Inventory (SIMULATED, lead time 2 weeks, 95% service, 13-week holdout):** across 120 SKUs, the
  ensemble-driven order-up-to policy cut the holding+shortage cost proxy by **42.8%** and raised fill rate from
  **89.5% to 93.6%** versus a seasonal-naive-driven policy. The ranking holds for all 12 lead-time x
  service-level combinations. It still fell short of the 95% target (weekly stock-out rate 7.2%); the reasons are
  discussed in section 12.

**Recommendation.** Forecast store x category-type aggregates with SARIMA or DHR-ARIMA (interpretable,
calibrated). Forecast the SKU long tail with a global gradient-boosting model. Deploy the SARIMA + LightGBM
average as the default, and size safety stock from cross-validated forecast errors, not from raw demand
variability.

---

## 2. Business context

In fashion and general-merchandise retail, demand forecasting drives three decisions with very different horizons:
**buying** (months ahead: how much to order from vendors), **allocation** (weeks ahead: which store gets how much)
and **replenishment** (days to weeks ahead: when to reorder a continuing line). Errors are asymmetric. An
under-forecast means a lost sale and a disappointed customer (in value fashion, the customer simply buys elsewhere).
An over-forecast means capital tied up in stock, warehouse space and, for seasonal fashion, end-of-season markdowns.
Value-fashion formats like Zudio and premium-department formats like Westside (Trent Limited) face both: a
fast-moving core of basics that replenishes continuously, and a long tail of size/colour options with intermittent
demand. That is exactly the aggregate vs SKU split studied here, although **no Trent data is used and no claim about
Trent's operations is made.**

What this project demonstrates for such a business: (1) how to choose a forecasting method per *type* of series
with evidence; (2) how to evaluate forecasts honestly (time-based backtests, significance tests, interval
calibration); (3) how forecasts become inventory parameters and what that is worth under clearly stated
assumptions.

---

## 3. The dataset

### 3.1 Choice, source, licence
| Item | Detail |
|---|---|
| Name | M5 Forecasting - Accuracy (Walmart), 2020 competition by the University of Nicosia (MOFC) and Walmart |
| Obtained from | the public mirror used by Nixtla's `datasetsforecast` package (`github.com/Nixtla/m5-forecasts`, `datasets/m5.zip`), so no Kaggle login is needed |
| Integrity | SHA-256 `cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774` (checked by `src/data/download.py`) |
| Licence / use | released for the M5 competition for research and educational use; this repository does **not** redistribute the raw data (download script only) and commits only small derived aggregates |
| Citation | Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). *M5 accuracy competition: Results, findings, and conclusions.* International Journal of Forecasting, 38(4), 1346-1364. |

**Candidates considered (in the required order):** (a) M5 via `datasetsforecast`: accepted (the raw URL was
blocked by the build sandbox's proxy, and the same file was obtained by a sparse git clone of the same public
repository); (b) UCI Online Retail II: reachable, but not needed and weaker (a single UK web shop, about 2 years,
transaction-level cleaning, no price/calendar tables); (c) Rossmann/Favorita: needed Kaggle credentials, which were
not available; (d) not needed.

**Selection criteria:** ≥ 2 years (5.3 years: pass); regular axis (0 calendar gaps, 0 missing cells: pass);
20+ series (30,490: pass); real demand (recorded unit sales: pass); usable licence (research use: acceptable).

### 3.2 Schema
- `sales_train_evaluation.csv` (30,490 x 1,947): `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, then
  `d_1 … d_1941` = units sold per day. The mirror drops Kaggle's redundant `id` column, which we rebuild.
- `calendar.csv` (1,969 x 13): date, Walmart week `wm_yr_wk`, weekday, month, year, up to two events per day with
  their type (Sporting / Cultural / National / Religious), and SNAP flags per state (days on which food-assistance
  benefits are paid).
- `sell_prices.csv` (6,841,121 x 4): weekly shelf price per store x item x week. A row exists only when the item
  is on sale.

### 3.3 Validation (`reports/data_profile.json`)
No missing sales cells, no duplicate ids or price keys, no negative sales or prices, no calendar gaps. 68.0% of
daily item x store cells are zero. The five days with total sales < 10% of the median are all 25 December (stores
closed).

### 3.4 Cleaning and construction decisions
| Decision | Why | Alternative rejected |
|---|---|---|
| Weekly buckets = Walmart weeks (Sat-Fri); 277 complete weeks; final 2 days dropped | a partial week would look like a demand collapse | calendar ISO weeks (would split Walmart's own week) |
| "Not listed" (no price row) = **no data**, not zero demand | verified: 0 units were sold in unlisted weeks; launch zeros would bias models to 0 | treating all zeros as demand |
| SKU series start at first listed week; eligibility ≥ 200 listed weeks, no gaps | guarantees ≥ 2 seasonal cycles before the first origin | include new launches (cold start is a separate problem) |
| Keep Christmas closures and level shifts | real calendar/business effects, not errors | capping/deleting (would erase real behaviour) |
| Keep suspected stock-out zero runs | true demand is unknown; imputing would invent data | imputation (needs inventory records we don't have) |

### 3.5 Limitations of the data for real retail demand
No apparel; sales ≠ demand (stock-outs unobserved: 59 of 120 sampled SKUs show ≥ 4-week zero runs despite
normally selling ≥ 5 units); no promotion flags (prices change in only 1.4% of SKU-weeks); no lead times, costs or
inventory positions; a US-specific calendar.

---

## 4. Problem definition
*(Written and committed before any model code; see `DECISIONS.md` D-08 to D-16.)*

| Decision | Choice | Why | Alternatives |
|---|---|---|---|
| Target | units | inventory is planned in units; revenue mixes in price | revenue |
| Granularity | weekly | removes the day-of-week pattern (Saturday +21%, Wednesday −13% vs the average day); zero share of listed SKU-weeks drops to 23.9%; matches weekly ordering | daily (noisy, 68% zeros), monthly (≈ 64 points, blurs holiday weeks) |
| Horizon | 13 weeks | a merchandise-planning quarter; its first 1-5 weeks feed replenishment | 4 weeks (M5), 26 weeks |
| Series | 211 across 6 levels, SKUs stratified by demand class | enables the "which model for which series" analysis | random SKU sample (51% smooth, few erratic) |
| Protocol | expanding-window rolling origin: 4 CV folds + 13-week holdout; 2 extra pre-CV origins for LightGBM tuning only | uses ~1 year of out-of-sample tests across seasons; tuning never touches CV or holdout | random split (leaks the future: a lag-52 feature would contain the test value), single split (fragile) |
| Primary metric | WAPE (volume-weighted), MASE for cross-series ranking | scale-free, defined with zeros, error weighted by volume | MAPE (undefined/infinite at 0 actuals) |
| Baselines | naive, seasonal naive, 4-week moving average | a model that cannot beat "same week last year" is not worth its maintenance cost | none |

**Metric definitions.** WAPE = Σ|y − ŷ| / Σy. MASE = MAE / (in-sample MAE of seasonal naive with lag 52), so MASE < 1
means better than seasonal naive's in-sample error. Bias = Σ(ŷ − y) / Σy (+ means over-forecast, i.e. overstock;
− means under-forecast, i.e. stock-outs). sMAPE = mean(2|y − ŷ| / (|y| + |ŷ|)): reported only for completeness,
because it is asymmetric and unstable near zero. Per-series metrics are summarised by the **median across series**
(robust) and **pooled** (volume-weighted) within a group; MASE is averaged across series.

---

## 5. Exploratory analysis

![Total and categories](reports/figures/01_total_and_categories.png)

**Trend and level shifts.** Average weekly total units: 184k (2011), 231k (2012), 252k (2013), 251k (2014), 265k
(2015) and 283k in the pre-holdout weeks of 2016. *Meaning:* there is a strong trend with a changing slope, so a
model must follow the level; this is why seasonal naive fails (section 9).

![STL](reports/figures/02_stl_total.png)

**STL decomposition** (robust LOESS, period 52) was preferred to classical decomposition because it lets the
seasonal shape evolve and down-weights outliers. The STL remainder flags 38 pre-holdout weeks with |robust z| > 3.5,
but they come in long runs (May-September 2012 is all positive), which is a **level/seasonal-amplitude shift**, not
isolated bad points. *Decision:* leave them (`reports/tables/outliers_total.csv`).

![Calendar effects](reports/figures/03_calendar_effects.png)

**Calendar effects.** Day-of-week index: Saturday 1.21, Sunday 1.20, Wednesday 0.88 (absorbed by weekly
bucketing). Month index ranges from 0.96 (November) to 1.04 (August). Event weeks vs the average of their
neighbouring weeks:

{{events}}

**SNAP (food-stamp days):** FOODS daily sales on SNAP vs non-SNAP days.

{{snap}}

*Meaning:* SNAP days are in the first half of every month, so weekly data contains a **~4.3-week cycle** that a
52-week seasonal model cannot represent. It shows in the ACF of differenced sales (lag 2 = −0.59, lag 4 = +0.65).

**Prices / promotions.** Prices change in only 1.4% of sampled SKU-weeks (median change 10.6%). The correlation
between weekly log-price change and log-units change is −0.03 (OLS slope −0.14). There are no promotion flags, so
promotion effects cannot be modelled (a limitation).

![ACF/PACF](reports/figures/04_acf_pacf_total.png)

**ACF/PACF.** Level: slow decay (lag 1 = 0.85, lag 13 = 0.71), so there is a unit root/trend. After d = 1: lag 52
= +0.71 (annual cycle) plus the lag 2/4 monthly pattern. After D = 1 only: lag 1 = 0.90 (trend remains). After d = 1
and D = 1: small negative spikes at lag 1 (−0.20) and lag 52 (−0.10), suggesting MA(1) x seasonal MA(1) terms
(an "airline"-type structure) and no strong over-differencing.

![Characteristics](reports/figures/05_series_characteristics.png)

**Series characterisation** (`reports/series_characteristics.csv`, pre-holdout data only). All 91 aggregates are
*smooth* with strong trend (median F_T 0.74-0.92 by level) and moderate-to-strong seasonality (median F_S
0.53-0.75). SKUs have median F_T 0.14, F_S 0.21, 24% zero weeks and CV ≈ 1.0. **This predicts, before any
modelling, that model choice should differ between aggregates and SKUs.**

**Zeros and censoring.** 59 of 120 SKUs have a run of ≥ 4 zero weeks despite selling ≥ 5 units in a typical
non-zero week: suspected stock-outs. Observed sales therefore understate demand for those items; we keep the data
and state the limitation.

---

## 6. Stationarity and decomposition: implications for model choice

ADF (H0: unit root) and KPSS (H0: stationary) test opposite hypotheses. Agreement is decisive; disagreement
signals trend-stationarity, breaks or low power. Results on the 21 total/category/department/store series
(pre-holdout):

{{stationarity}}

*Meaning:* **d = 1** is needed and sufficient for the non-seasonal part (21/21 stationary after one difference).
Seasonal differencing alone does not achieve stationarity (1/21), and these tests cannot decide D. D is therefore
chosen with the seasonal-strength rule (D = 1 if F_S > 0.64) used by `auto.arima`/statsforecast. Variance does
**not** grow with level on the total (yearly standard deviation 16k-19k while the mean rose 54%), so no log/Box-Cox
transform is used by default; this was confirmed out of sample in section 8.

---

## 7. Models
All models see exactly the same training window at each origin; forecasts are clipped at 0 (identically for all).

| Model | Theory in one line | Key assumptions | Setup / tuning | Why included |
|---|---|---|---|---|
| Naive | ŷ = last value | random walk | none | floor benchmark |
| SeasonalNaive | ŷ = same week last year (lag 52) | stable seasonality, no growth | none | the "planner's rule", must be beaten |
| MovingAverage | mean of last 4 weeks | locally flat level | window 4 (≈ 1 month) | smoothing without seasonality |
| SES | exponentially weighted level | level only | α by likelihood (statsforecast) | smoothing baseline |
| ETS | state-space level/trend/season with additive or multiplicative errors | stable components, Gaussian errors | candidates A/M error, N/Ad trend, N/A/M season (M only if y > 0), chosen by AICc counting 51 free seasonal states; statsmodels `ETSModel` because statsforecast's AutoETS drops seasonality for s > 24 | classical seasonal workhorse |
| SARIMA | ARIMA on (seasonally) differenced data with lag-52 AR/MA terms | linear, Gaussian, constant parameters | statsforecast AutoARIMA (Hyndman-Khandakar stepwise AICc search; d by KPSS, D by seasonal strength; CSS approximation in the search, as R's auto.arima does by default for long/high-period series) | the highlighted model |
| DHR-ARIMA | Fourier terms (period 52.18) + Christmas/Thanksgiving/Easter/Super Bowl week dummies + SNAP days + ARIMA errors | smooth seasonality, linear driver effects | K ∈ {2,4,6,8,10} chosen per series by AICc on pre-CV data; ARIMA orders re-selected at each origin | handles s = 52.18 and known drivers ("SARIMAX" family) |
| Prophet | piecewise-linear trend + yearly Fourier + holiday-week effects + SNAP regressor | additive components | defaults; holidays as weekly events | popular, holiday-aware |
| LightGBM | one global gradient-boosted tree model for all 211 series; direct multi-horizon | relationships shared across series and time | target and lags scaled by the trailing-52-week mean; features in section 7.1; tuned on 2 pre-CV origins only | cross-series learning, exogenous drivers, nonlinearity |
| Ensemble | mean of the best statistical model (by CV: SARIMA) and LightGBM | errors not perfectly correlated | components fixed using CV results only | forecast combination is usually robust |

### 7.1 LightGBM design (leakage control)
- **Direct, not recursive.** One row per (series, origin T, horizon h ≤ 13) with h as a feature. Recursive
  forecasting feeds predictions back as lags, so errors compound over 13 steps and future exogenous inputs must be
  simulated; the direct strategy avoids both, at the cost of more training rows (327k-541k).
- **Features from y use only weeks ≤ T:** last 4 values, rolling means (4/8/13/26/52), rolling std (4/13/52), zero
  shares, 26-week slope, year-over-year ratio, seasonal lags y[T+h−52] and y[T+h−104] (≤ T because h ≤ 13),
  series age. **Known-in-advance target-week features:** week of year, month, holiday/event counts, SNAP days, and
  the planned shelf price relative to the trailing average (an assumption, ablated below). Categorical IDs: level,
  category, department, store.
- **Leakage test** (`tests/test_no_leakage.py`): build the features, overwrite every future value with garbage,
  rebuild, and assert the features are identical. Also assert that no training target lies after the origin. Passes.
- **Training labels** are winsorised at their 99.9th percentile (a few SKUs with near-zero trailing scale gave
  scaled targets > 20).
- **Tuning** (12-point grid, mean MASE on the two tuning origins only):

{{lgbm_tuning}}

The spread across all 12 points is small (0.708-0.736), so hyper-parameters matter little.
- **Prediction intervals:** empirical quantiles of scaled errors from *earlier* origins only (split-conformal
  style), by group and horizon bucket.

---

## 8. SARIMA deep dive (TOTAL series)

**Notation.** SARIMA(p,d,q)(P,D,Q)[s]: p AR lags, d ordinary differences, q MA lags; P, D, Q the same at the
seasonal lag s = 52. Orders were **identified using only weeks 0-211** (CV fold 1 training data) and evaluated on
all later windows with parameters re-estimated, so the order choice never saw its test data.

1. **Differencing:** `ndiffs` (KPSS) = 1, `nsdiffs` (seasonal strength 0.684 > 0.64) = 1, so d = 1, D = 1.
   Over-differencing would show as a strongly negative lag-1 residual autocorrelation; none is seen.
2. **Order search:** AICc over p ≤ 3, q ≤ 2, P ≤ 1, Q ≤ 1 (48 models, 50 s). AICc winner: **(3,1,2)(0,1,1)[52]**.
   Information criteria measure in-sample fit, so the top 6 were re-checked out of sample:

{{sarima_grid_top}}

   The top six are within ~4 AICc units, and CV ranks them differently (MASE 0.300-0.322). Many orders are
   equally good and **the result does not hinge on the exact order**. That justifies the automatic stepwise search
   for the 211-series benchmark: on the same window it chose (3,1,1)(1,1,1) with the CSS approximation (2.1 s) and
   (3,1,0)(0,1,0) with exact ML (17.5 s).
3. **Transformation:** log vs none, same orders, CV MASE:

{{sarima_transform}}

   The difference (0.306 vs 0.309 mean CV MASE) is negligible and not consistent across folds, so **no transform**
   is used (consistent with the constant variance in section 6).
4. **Residual diagnostics** (model fitted on weeks 0-263):

![SARIMA residuals](reports/figures/09_sarima_residual_diagnostics.png)

{{sarima_lb}}

   *White-noise residuals* would mean the model has extracted all linear, predictable structure. Here Ljung-Box
   **rejects** at lags 8 (p = 0.003) and 13 (p = 0.033) but not at 26 (0.14) or 52 (0.08). The visible culprit is
   a lag-2 residual autocorrelation of −0.16, a plausible remnant of the monthly SNAP/payday cycle (half a ~4.3-week
   cycle apart gives a negative correlation). **The model captures the annual structure but not all short-cycle
   structure.** Normality: skew 0.23, excess kurtosis 0.68, Jarque-Bera p = 0.053, Shapiro p = 0.26, so roughly
   Gaussian with slightly heavy tails.
5. **Accuracy and interval calibration:**

![SARIMA windows](reports/figures/09b_sarima_total_windows.png)

   WAPE 2.1-3.3% per 13-week window; bias −1% to −2% in CV (slight under-forecast during growth) and +0.1% on the
   holdout. Of 65 forecast weeks, **95.4% fell inside the nominal 80% interval and 100% inside the 95% interval**,
   so the TOTAL intervals are **too wide (conservative)**: σ² absorbs the 2012 shift and the unmodelled monthly
   cycle. Across all 211 series, AutoARIMA's intervals are close to nominal (section 9.4).
6. **The seasonal-period problem.** (i) A year is 52.18 weeks, so a lag-52 term drifts ~1 day per year and
   holidays move relative to "same week last year"; (ii) the SARIMA state vector grows with s, and statsmodels'
   SARIMAX with s = 52 did not finish one fit within 10 minutes in this environment (with other jobs running),
   whereas statsforecast's implementation takes about 1-2 s; (iii) each seasonal parameter is estimated from only
   about 4 annual repetitions. **Handling:** SARIMA with s = 52 is kept as the benchmark model, and
   **DHR-ARIMA** (Fourier with period 52.18 + holiday dummies + SNAP + ARIMA errors) is added as the principled
   alternative. Monthly granularity was rejected (≈ 64 points; blurs holiday weeks).
7. **What AutoARIMA actually chose across the benchmark:** D = 1 in 37.4% of aggregate fits but only 1.7% of SKU
   fits; d = 1 in 97.4% of aggregate fits and 69.3% of SKU fits. On SKUs the seasonal structure is too weak for
   seasonal differencing to pay off.

---

## 9. Results

### 9.1 Overall leaderboards (211 series)
**CV folds** (used for every model-selection decision):

{{lb_cv}}

**Holdout** (final, untouched):

{{lb_ho}}

*Reading:* median WAPE is dominated by the 120 SKUs, where 40-70% weekly errors are normal; pooled WAPE is
volume-weighted. The Ensemble leads both. SARIMA, DHR-ARIMA and LightGBM form the next group. Seasonal naive is
last, with MASE > 1.

### 9.2 Aggregates vs SKUs
Mean MASE (CV):

{{lb_cv_group_mase}}

Pooled WAPE (CV):

{{lb_cv_group_pooled}}

Median bias (CV):

{{lb_cv_group_bias}}

Holdout mean MASE / pooled WAPE:

{{lb_ho_group_mase}}

{{lb_ho_group_pooled}}

**WAPE improvements vs baselines (CV, from the tables above):** on aggregates, SARIMA's median WAPE is 7.9% vs
14.9% for seasonal naive (a 47% relative reduction) and 9.6% for naive (−18%); pooled WAPE 5.3% vs 10.2%. On SKUs,
LightGBM's median WAPE is 70.3% vs 93.4% for seasonal naive (−25%) and 83.3% for naive (−16%).

### 9.3 By hierarchy level and demand class
![By level](reports/figures/07_leaderboard_by_level.png)

Mean MASE by level (CV):

{{lb_cv_level}}

Median WAPE by level (CV):

{{lb_cv_level_wape}}

Mean MASE by level (holdout):

{{lb_ho_level}}

![By class](reports/figures/08_mase_by_class_heatmap.png)

Mean MASE by demand class (CV, then holdout):

{{lb_cv_class}}

{{lb_ho_class}}

By seasonal-strength band (CV):

{{lb_cv_seas}}

By trend-strength band (CV):

{{lb_cv_trend}}

### 9.4 Prediction-interval coverage (share of actuals inside the interval)

![Coverage](reports/figures/10_interval_coverage.png)

{{coverage}}

SARIMA, DHR-ARIMA, LightGBM (empirical intervals) and seasonal naive are near nominal. **ETS under-covers badly on
aggregates (≈ 50% for a nominal 80%)** and so does **Prophet (≈ 58%)**; ETS over-covers on SKUs. Using such
intervals for safety stock would be dangerous, so calibration must be measured, not assumed.

### 9.5 Statistical comparison
Paired bootstrap across series of the mean CV MASE difference (A − B; negative = A better; 5,000 resamples, series
as the resampling unit):

{{bootstrap}}

Per-series Diebold-Mariano tests (absolute-error loss, HLN correction, autocovariance to lag 12, 65 forecast
weeks per series):

{{dm}}

*Meaning:* group-level differences between the top models are real but small; at the level of a single series,
most differences are within noise. On the TOTAL series alone, SARIMA vs DHR-ARIMA (p = 0.10) and Ensemble vs SARIMA
(p = 0.21) are not significant; SARIMA beats seasonal naive (p = 0.044) and ETS (p = 0.030).

### 9.6 Fold-by-fold stability
Mean MASE per origin:

{{fold_mase}}

Median WAPE per origin:

{{fold_wape}}

### 9.7 LightGBM: what it uses, and the price-assumption ablation

{{lgbm_imp}}

{{lgbm_ablation}}

Without the target-week price features, LightGBM's mean MASE moves by less than 0.01 in every group/period, so
**no conclusion depends on the "planned price is known" assumption.** (The "with_price" numbers reproduce the main
run exactly, an incidental determinism check.)

---

## 10. Why each model works where it does (and not elsewhere)

![SARIMA relative error](reports/figures/11_sarima_relative_error_vs_characteristics.png)

Binned win rates and median MASE by series characteristic (CV):

{{why_bins}}

By class:

{{why_class}}

![SARIMA vs LightGBM](reports/figures/12_sarima_vs_lightgbm.png)
![Winners](reports/figures/13_winner_share_by_class.png)

Number of series won (lowest CV MASE) by class:

{{why_winners}}

### 10.1 SARIMA
- **Wins where** series are aggregates (no zeros), with strong trend and seasonality and long, mostly stable
  history. Its win rate vs LightGBM rises from 38% (F_S ≤ 0.2) to 66% (F_S > 0.6), and vs ETS from 50% to 85%. On
  zero-free series it beats LightGBM 59% of the time; with trend strength > 0.8, 59% (and ETS 82%). It is the best
  model at department level (0.519) and the best single model on aggregates (0.636), with calibrated intervals.
- **Loses where** demand is SKU-level and zero-inflated: with zeros it beats LightGBM on only 32-48% of series.
  ARIMA assumes continuous, roughly Gaussian errors and linear autocorrelation, but weekly SKU demand with 24-37%
  zeros has little such structure (median F_S ≈ 0.2), and AutoARIMA selects seasonal differencing on only 1.7% of
  SKU fits. On 4 strongly seasonal SKUs it was the worst of the top models (mean MASE 1.03 vs Prophet 0.79), a
  small-n hint that lag-52 terms estimated from ~4 noisy repetitions overfit noise.
- **Contradiction with the textbook, investigated.** One would expect SARIMA's edge over *seasonal naive* to grow
  with seasonal strength. It does not (it wins 83-92% in every seasonal band) because seasonal naive's problem here
  is trend, not seasonality: with growth, last year's level is always too low (median aggregate bias −11.6%).

### 10.2 LightGBM (global)
- **Wins where** there are many related, noisy series: best single model on SKUs (0.733), especially erratic (0.711)
  and smooth (0.659) SKUs. It pools patterns across 211 series, uses exogenous drivers (SNAP, events, price), and
  makes no distributional assumption about zeros.
- **Loses where** series are strongly trending aggregates (ΔMASE −0.050 in SARIMA's favour, significant). Trees
  predict leaf averages and cannot extrapolate beyond the training range; the trailing-mean scaling helps, but each
  aggregate's specific autocorrelation is better captured by a per-series model. It is also sensitive to the
  fold: best single model in cv1 (0.669), worst of the top four in cv4 (0.741).
- **Leakage-prone design was avoided** by construction and by test (section 7.1).

### 10.3 DHR-ARIMA, ETS, Prophet, baselines
- **DHR-ARIMA** ties SARIMA overall and wins most store and total series (store 0.431 vs SARIMA 0.466; total 0.326
  vs 0.331): explicit SNAP and holiday regressors pay off on large FOODS-heavy aggregates. It has fewer parameters
  for the annual cycle than SARIMA's lag-52 structure and uses the correct 52.18-week period.
- **ETS** is significantly worse than SARIMA on aggregates (−0.089) and badly calibrated there (50% coverage of
  80% intervals), because it must carry 52 seasonal states. It is the **best model on lumpy SKUs (0.739)**, where
  heavy smoothing of a noisy level is what matters.
- **Prophet** is mid-table overall (0.774) with the most negative SKU bias (−12.9%), since it ignores short-term
  autocorrelation. But it has the **best mean MASE in the strong-seasonality band (0.699, 41 series)**, driven by
  store x department and the few strongly seasonal SKUs. Smooth Fourier seasonality is robust when the seasonal
  pattern is strong but noisy.
- **Naive / moving average** win a share of intermittent and lumpy SKUs (naive wins 8 of 30 intermittent SKUs):
  when demand is mostly noise, the simplest level estimate is hard to beat.
- **Seasonal naive** is last (MASE 1.020) because of growth.

### 10.4 Where the choice does not matter
The CV-best model is also the holdout-best model on only **12.8%** of series. A depth-3 decision tree predicts the
per-series winner with **20.4%** accuracy vs **18.5%** for always guessing the most common winner. For SARIMA vs
LightGBM, 75/91 aggregates and 99/120 SKUs show no significant per-series DM difference. **Choosing a different
model for every series is mostly fitting noise.** Robust choices are made per *group*, and combining models (the
Ensemble) captures most of the benefit.

### 10.5 Decision guide
| Given a series that is… | prefer | because (evidence in this study) |
|---|---|---|
| an aggregate (store/department/category), zero-free, trend & seasonal strength > 0.5 | **SARIMA** or **DHR-ARIMA** | best single models on aggregates (0.636 / 0.646), significantly better than LightGBM (−0.050) and ETS (−0.089), calibrated intervals |
| a large aggregate with known calendar drivers (SNAP, moving holidays) | **DHR-ARIMA** | best on total/category/store; correct 52.18-week period |
| one of many related SKU series with zeros and weak seasonality | **global LightGBM** | best single model on SKUs (0.733); still 0.741 without price features |
| lumpy (ADI ≥ 1.32 and CV² ≥ 0.49) | **ETS/SES** or the ensemble | ETS best on lumpy SKUs (0.739) |
| strongly seasonal but noisy | **Prophet** is worth a try | best in the F_S > 0.6 band (0.699) |
| anything, if you can deploy only one approach | **Ensemble (SARIMA + LightGBM)** | best overall in CV (0.681) and holdout (0.668); better than SARIMA on 68% of series |
| strongly trending | **not** seasonal naive | MASE 1.11 on aggregates, −11.6% bias |

---

## 11. Final forecast

Each series is forecast with the model that had the best mean CV MASE at its level (DHR-ARIMA for
total/category/store, SARIMA for departments, Ensemble for store x department and SKUs), refit on all 277 weeks.
13 weeks: 2016-05-21 to 2016-08-13. All candidates' forecasts are in `outputs/forecasts.csv` (flag `recommended`).

![Final forecast](reports/figures/14_final_forecast.png)

{{headline}}

**What it says:** about 4.27 M units over the next 13 weeks for the total, +19.5% vs the same weeks of 2015. That
is consistent with the momentum in the data (the last 13 actual weeks were already +16.2% year on year) and with
the other strong models (SARIMA, ETS and the Ensemble give similar weekly levels of 289k-348k). FOODS drives the
growth (+22.9%); HOUSEHOLD +9.6%, HOBBIES +7.6%. Weekly peaks fall in the first week of each month (SNAP) and in
early August.
**What it does not say:** nothing about future promotions, price changes (prices are held constant), competitor
actions, supply problems or store openings. The intervals only reflect variability seen in the past. At aggregate
level the CV evidence suggests ~2-4% WAPE on a 13-week quarter for the total and roughly calibrated 80% intervals
for SARIMA/DHR.

---

## 12. Inventory optimization (SIMULATED)

### 12.1 Method
- **Policy:** periodic review every R = 1 week, order-up-to level S_t, lost sales (customers do not wait), lead
  time L. S_t = forecast demand over the protection interval L + R + safety stock.
- **Safety stock** SS = z · σ · √(L + R); **reorder point** (continuous-review view) ROP = forecast over L +
  z · σ · √L; z = Φ⁻¹(service level), e.g. 1.645 for 95%.
- **σ = RMSE of the model's own CV forecast errors at horizons 1-4** (16 errors per series), **not** the standard
  deviation of demand. The forecast already explains the predictable part (trend, seasonality, SNAP), so only the
  unexplained part needs a buffer. A better forecaster is rewarded twice: better cycle stock and a smaller buffer.
- **Assumptions (stated in `config/config.yaml`):** lead time L ∈ {1,2,3,4} weeks (M5 has none), service level ∈
  {90, 95, 98}%, holding cost 1 per unit-week, shortage cost 9 per lost unit (critical ratio 0.9). Default
  L = 2, SL = 95%.
- **Back-test:** the 13 holdout weeks. Each week the forecast is updated with the latest actual (rolling origin,
  horizons 1..L+1) without re-estimating parameters, as a weekly replenishment run would do (SARIMA via
  `forward()`, LightGBM via refreshed features). Policies are driven by naive, seasonal naive, SARIMA, LightGBM and
  the Ensemble, plus "Ensemble + demand-sigma" (same forecasts, σ = demand standard deviation) to test the σ choice.
  Initial stock = SS plus pipeline orders equal to the first forecasts, identical across policies.

### 12.2 Results (L = 2 weeks, 95% service)

{{inv_default}}

**SKUs (120):** the Ensemble policy's cost proxy is **42.8% lower** than seasonal naive's, with fill rate **93.6% vs
89.5%** and 44% less average stock. LightGBM −42.5%, SARIMA −38.9%. Naive reaches the highest fill rate (95.0%)
but holds 41% more stock than the Ensemble policy.
**Store x department (70):** Ensemble −55.4%, SARIMA −55.8%, LightGBM −46.4% vs seasonal naive, whose −11%
growth bias produces a 24.9% weekly stock-out rate.
**σ choice:** sizing safety stock on demand variability instead of forecast error raises store x department safety
stock by 34% (92k vs 69k units) and average stock by 27%, with a marginal fill gain (99.8% vs 99.4%) and a 12%
higher cost proxy.

### 12.3 Sensitivity (cost proxy vs the seasonal-naive policy, %)

{{inv_sens}}

![Inventory sensitivity](reports/figures/15_inventory_sensitivity.png)

The ranking is identical in all 24 group x lead-time x service-level cells: SKU savings 37-43%, store x department
43-66% (SARIMA and Ensemble are the strongest at store x department, the Ensemble and LightGBM at SKU).

### 12.4 Honest reading
1. Much of the gain vs seasonal naive comes from that baseline's growth bias in a quarter that was +16% YoY.
   Against the naive policy the gain is smaller (Ensemble cost 21% below naive at SKU level: 35.2k vs 44.5k).
2. **The service target is under-delivered at SKU level:** at 95%, model-driven policies stock out in 6.1-7.9% of
   SKU-weeks instead of 5%. Reasons: σ comes from only 16 errors per series; the √(L + R) rule assumes independent
   weekly errors (multi-step errors are positively correlated); censored zero runs understate variability; and
   13 weeks is a short window. With real data, one would estimate σ for the protection interval directly from
   rolling backtests and check empirical service by segment.
3. These are **simulated** outcomes under assumed costs and lead times, not realised savings.

### 12.5 At-risk series and actions

{{risk_counts}}

Top stock-out risks (simulated fill < 90% or CV bias < −10%):

{{risk_stock}}

Top overstock risks (> 3 weeks of cover or CV bias > +10%):

{{risk_over}}

**Merchandising/replenishment reading:** stock-out risks are mostly lumpy/erratic SKUs whose forecasts run low.
Action: raise their safety factor, check for unrecorded stock-outs, and set presentation minimums. Overstock risks
are slow intermittent SKUs where one unit is many weeks of demand. Action: a min/max or order-multiple rule, an
intermittent-demand method (Croston/TSB), or range rationalisation.

---

## 13. Limitations and threats to validity
1. **Public, non-apparel US data.** Methods transfer; numbers do not. Fashion adds short life-cycles, size curves,
   markdowns and cold starts.
2. **Censoring.** Sales ≠ demand; suspected stock-outs in 59/120 SKUs bias forecasts and σ downward.
3. **No promotions** and few price changes; no causal price/promo model.
4. **Inventory is simulated** with assumed lead times, costs, review period and initial stock, on a 13-week window.
5. **Small differences.** Top models differ by ~0.01-0.05 MASE; most per-series differences are not significant.
6. **Sample design.** 120 SKUs stratified by class (not representative of SKU volume shares); SKUs need ≥ 200
   listed weeks, so new items are excluded.
7. **Characteristics** were computed on all pre-holdout data, so for the earliest CV folds they include some
   later weeks. They are used only for *explaining* results, never as model inputs.
8. **Computation shortcuts:** AutoARIMA's CSS approximation; DHR's K fixed per series from pre-CV data; ETS
   heuristic seasonal initialisation (which probably explains part of its interval under-coverage).
9. **Single dataset period** (2011-2016), including a strong-growth phase that penalises seasonal naive.

---

## 14. Conclusions and next steps with real company data
**Conclusions.** (1) SARIMA earns its reputation on seasonal, trending aggregate series, where it is the best
single model with calibrated intervals; a Fourier/holiday regression with ARIMA errors is equally good. (2) On
noisy SKU demand a global LightGBM is better, and on lumpy items simple smoothing competes. (3) Combining SARIMA
and LightGBM is the most robust single choice. (4) Seasonal naive is a misleading baseline in a growth period.
(5) Forecast-error-based safety stock is cheaper than demand-based safety stock, and accuracy gains translate into
large simulated inventory improvements, but service targets must be verified, not assumed.

**With real Trent-style data I would:**
1. **Hierarchical reconciliation** (MinT / bottom-up / middle-out) so that store x category x SKU forecasts add up,
   using the aggregate-strength of SARIMA/DHR and the SKU-strength of the global model.
2. **Promotion and markdown modelling**: promo calendars, discount depth, display/space, and price elasticity per
   category; causal features in the global model.
3. **New-product cold start**: attribute-based models (fabric, fit, colour, price band) and analogue products.
4. **Censored-demand correction** using inventory/availability records (unconstrain sales before training).
5. **Probabilistic forecasting** (quantile LightGBM / conformal intervals) feeding service-level-driven
   inventory, with σ estimated per protection interval.
6. **Size-curve and allocation** models for fashion; Indian calendar effects (Diwali, wedding season, EOSS).
7. **Deployment & monitoring**: a weekly pipeline, forecast-value-added tracking vs naive, drift alerts on bias,
   and a champion/challenger set-up.

---

## 15. Appendix

### 15.1 Environment

{{env}}

### 15.2 Reproduction
```bash
git clone <repo> && cd RetailPulse
make setup          # venv + pinned requirements
make all            # data -> eda -> train (~45 min, 4 cores) -> eval -> why -> forecast -> inventory -> export -> notebooks -> test
make app            # Streamlit dashboard
python -m src.analysis.sarima_deep_dive && python -m src.models.lgbm_ablation && python -m scripts.render_report
```
Seeds are fixed (42). Re-running evaluation, the why-analysis, EDA and the SARIMA deep dive inside the notebooks
regenerated every table byte-for-byte (only wall-clock timing fields changed).

### 15.3 Tests
`pytest` = **26 passed**: metric correctness (WAPE, MASE incl. fallback, bias sign, sMAPE, coverage, DM test),
time-split ordering (non-overlapping, expanding, holdout last, tuning before CV), **no-leakage** (features unchanged
when all future values are corrupted; no training target after the origin; seasonal lags in the past), and
inventory formulas (z-values, SS = z·σ·√(L+R), ROP, order-up-to, simulator conservation and edge cases).

### 15.4 Glossary
- **ACF / PACF**: correlation of a series with its own lags / the same after removing shorter lags.
- **ADF / KPSS**: stationarity tests with opposite null hypotheses (unit root vs stationary).
- **ADI / CV²**: average interval between non-zero demands / squared coefficient of variation of non-zero sizes.
- **AICc**: Akaike information criterion with a small-sample correction; lower = better fit-complexity trade-off.
- **ARIMA / SARIMA**: autoregressive integrated moving-average model / with seasonal terms at lag s.
- **Bias**: Σ(forecast − actual)/Σ actual.
- **Box-Cox / log transform**: variance-stabilising transformations.
- **Censoring**: observed sales capped by available stock.
- **Cycle service level**: probability of no stock-out in a replenishment cycle; **fill rate**: share of demand served.
- **Diebold-Mariano test**: tests whether two forecasts have equal expected loss.
- **DHR (dynamic harmonic regression)**: Fourier terms as regressors with ARIMA errors.
- **Direct vs recursive multi-step**: one model per (origin, horizon) vs feeding forecasts back as inputs.
- **ETS**: Error-Trend-Seasonal exponential smoothing state-space models.
- **Fourier terms**: sin/cos pairs that represent a smooth periodic pattern.
- **Global model**: one model trained on many series.
- **Holdout**: final test window never used for any decision.
- **Ljung-Box test**: tests whether residual autocorrelations up to lag k are jointly zero.
- **MASE**: MAE scaled by the in-sample seasonal-naive MAE.
- **Order-up-to (base-stock) policy**: each review, order enough to raise the inventory position to S.
- **Protection interval**: L + R; the period an order must cover.
- **Reorder point**: stock level that triggers an order in continuous review.
- **Rolling-origin CV**: repeated forecasts from successive origins, each trained only on the past.
- **Safety stock**: buffer for forecast error over the protection interval.
- **Seasonal / trend strength**: share of variance explained by the STL seasonal / trend component.
- **SNAP**: US Supplemental Nutrition Assistance Program; benefit payment days lift food sales.
- **STL**: Seasonal-Trend decomposition using LOESS.
- **Syntetos-Boylan classes**: smooth, erratic, intermittent, lumpy (by ADI and CV²).
- **WAPE**: Σ|error| / Σ actual.
- **White noise**: uncorrelated, zero-mean residuals; no predictable structure left.

### 15.5 Interview Q&A (answers grounded in this project's results)
1. **Why WAPE and not MAPE?** MAPE divides by each actual, so it is undefined for the 23.9% of listed SKU-weeks with
   zero sales and explodes for small actuals. WAPE divides total absolute error by total volume: defined with
   zeros, scale-free, and weighted the way the business feels errors.
2. **Why MASE too?** WAPE cannot be averaged across series with very different noise levels. MASE scales each
   series by its own seasonal-naive error, so it can be averaged across 211 series; < 1 means better than
   seasonal naive's in-sample error.
3. **Why time-based CV instead of a random split?** A random split trains on the future, for example a lag-52
   feature would contain the test value. Rolling origin mimics deployment, and 4 folds + a holdout cover about a
   year of different seasons, so one lucky quarter cannot decide the winner.
4. **Why is seasonal naive a good baseline, and why did it do so badly?** It is free, needs no maintenance and is a
   common planning rule. Here demand grew strongly (184k to 283k units/week), so "same week last year" was
   systematically too low (median aggregate bias −11.6%; MASE 1.11). That tells you the dominant signal was trend.
5. **How did you choose d and D?** KPSS/ADF: all 21 aggregates became stationary after one difference (d = 1).
   Seasonal differencing by the seasonal-strength rule: F_S = 0.68 > 0.64 on the total, so D = 1. The ACF after
   both shows only small negative spikes (no over-differencing).
6. **What did the Ljung-Box test tell you?** Residuals of SARIMA(3,1,2)(0,1,1)[52] on the total are white noise at
   long lags (p = 0.14 at 26, 0.08 at 52) but not at short lags (p = 0.003 at 8, 0.033 at 13): a lag-2 spike
   consistent with the monthly SNAP cycle, which a 52-week model cannot capture. That motivated the DHR model with
   SNAP regressors.
7. **How did you pick SARIMA orders, and do you trust AICc?** A 48-model AICc grid on pre-CV data, then the top 6
   checked on CV. They were within ~4 AICc units and CV ranked them differently (MASE 0.30-0.32), so the exact order
   barely matters. AICc is an in-sample criterion, so I confirm out of sample.
8. **Where does SARIMA work and where not?** It works on zero-free aggregates with strong trend and seasonality
   (best single model on aggregates, MASE 0.636; beats LightGBM on 66% of strongly seasonal series). It does not
   work on zero-heavy SKUs (beats LightGBM on only 32-48% of series with zeros) because its Gaussian, linear
   assumptions don't fit intermittent counts.
9. **Why did LightGBM win on SKUs but lose on aggregates?** On SKUs it pools information across 211 series and uses
   SNAP/event/price features. On trending aggregates trees cannot extrapolate beyond training targets, and a
   per-series ARIMA captures each series' own dynamics better (significant ΔMASE −0.050 in SARIMA's favour).
10. **How did you prevent leakage in LightGBM?** Features from sales use only weeks ≤ origin, and the training set
    only includes rows whose target is ≤ origin. A unit test corrupts all future values and asserts the features
    are unchanged. Tuning used two origins before the first CV fold. Target-week price is an assumption, and
    removing it changed MASE by < 0.01.
11. **Is the ensemble really better or just lucky?** A paired bootstrap across series gives ΔMASE −0.020 [−0.030,
    −0.011] vs SARIMA, better on 68% of series, and it also leads on the holdout (0.668). The gain is real but
    small.
12. **Did your prediction intervals work?** For SARIMA, DHR-ARIMA and LightGBM, about 83-86% of actuals fell in the
    80% interval. ETS (50%) and Prophet (58%) were badly too narrow on aggregates. On the total series SARIMA's were
    too wide (95% inside the 80% interval).
13. **How is safety stock computed and why from forecast error?** SS = z·σ·√(L+R), with z = 1.645 at 95% and σ = the
    RMSE of CV forecast errors. Demand variability includes predictable seasonality and trend; using it raised
    store x department safety stock by 34% for a 12% higher cost proxy.
14. **What are the inventory results, and can you claim savings?** Simulated, at L = 2 and 95%: the ensemble policy
    cut the holding+shortage cost proxy by 42.8% vs a seasonal-naive policy across 120 SKUs (fill 93.6% vs 89.5%).
    These are not real savings: the lead time, costs and initial stock are assumptions, and the window is 13 weeks.
15. **What went wrong or surprised you?** Seasonal naive was the worst model; the per-series best model is unstable
    (CV and holdout agree on only 12.8% of series); ETS and Prophet intervals were badly under-calibrated; the SKU
    policies missed the 95% service target (6.1-7.9% stock-out weeks vs 5%).
16. **Why weekly and a 13-week horizon?** Weekly removes the day-of-week pattern and most zeros and matches weekly
    ordering; 13 weeks is a merchandise-planning quarter whose first weeks drive replenishment.
17. **Why s = 52 is a problem and what you did?** The year is 52.18 weeks, so holidays drift; s = 52 makes the
    state space large (statsmodels was impractically slow); and there are only ~4 cycles to estimate seasonal
    parameters. I added Fourier-based DHR with the correct period and holiday/SNAP regressors, which ties SARIMA
    overall and wins at total/store level.
18. **How would this change with Trent data?** Hierarchical reconciliation, promotion/markdown features, cold start
    for new styles, size curves, censored-demand correction from stock records, Indian calendar effects, and
    probabilistic forecasts feeding service-level-based inventory.
19. **How do you know the results are reproducible?** Seeds are fixed, versions are pinned, there is one `make all`,
    and re-running the evaluation, why-analysis, EDA and SARIMA deep dive in the notebooks regenerated all tables
    byte-for-byte.
20. **What is the single most important takeaway?** Choose the method by series *type* with evidence (SARIMA/DHR for
    stable seasonal aggregates, global ML for the noisy SKU long tail), combine them, and judge everything with
    time-based backtests, significance tests and calibration checks.
