"""Cells for notebooks 04-06 (results). Imported by scripts/build_notebooks.py."""
from scripts.build_notebooks import SETUP, code, md

# ------------------------------------------------------------------------------------------------ 04
NB04 = [
    md("""# 04 · Model benchmark

**Protocol (identical for every model):** 211 weekly series; 4 rolling-origin (expanding-window) CV folds, each with a 13-week test window, plus a final untouched 13-week holdout (2016-02-20 to 2016-05-14). Each model is refit at every origin on data up to that origin only. Forecasts are clipped at 0 for all models.

Training is expensive (about 45 min on 4 cores), so it runs via `make train`; `make eval` scores it. This notebook re-runs the **scoring** step from the cached forecasts and explains the results."""),
    code(SETUP),
    md("""## Models and why each is in the race
| Model | Idea | Assumptions | Should work when | Should fail when |
|---|---|---|---|---|
| Naive | next = last value | random walk | level changes unpredictably | strong seasonality |
| SeasonalNaive | next = same week last year | stable annual pattern, no growth | strong, stable seasonality | trend / growth, moving holidays |
| MovingAverage (4 wk) | mean of last 4 weeks | locally constant level | noisy, no structure | trend, seasonality |
| SES | exponentially weighted level | level only | noisy, slowly drifting level | trend, seasonality |
| ETS | level + (damped) trend + season states, chosen by AICc | errors ~ Gaussian, stable components | smooth, seasonal series | zeros (multiplicative not allowed), 52 seasonal states to initialise |
| SARIMA | ARIMA on differenced data incl. lag-52 terms (AutoARIMA) | linear, Gaussian, constant parameters after differencing | smooth aggregates with stable seasonal and trend dynamics | intermittent counts, short history, breaks, drivers it cannot see |
| DHR-ARIMA | Fourier (period 52.18) + holiday + SNAP regressors + ARIMA errors | smooth seasonality, linear effects | non-integer period, moving holidays, known drivers | sharp, irregular seasonal shapes |
| Prophet | piecewise-linear trend + Fourier + holidays + SNAP regressor | additive components, changepoints | multiple seasonalities, holidays | short-term autocorrelation, intermittent data |
| LightGBM (global) | one gradient-boosted tree model for all series, scaled target, lags + calendar + price | stationarity of relationships across series and time | many related series, nonlinear and exogenous effects | extrapolating trends beyond the training range |
| Ensemble | mean of the best statistical model (by CV) and LightGBM | errors not perfectly correlated | the two models fail differently | one model is much worse |"""),
    code("""import contextlib, io
from src.evaluation import evaluate
with contextlib.redirect_stdout(io.StringIO()):
    evaluate.main()
EV = J("reports/evaluation_summary.json")
print("ensemble components (chosen on CV only):", EV["ensemble_components"])
print("statistical models ranked by CV mean MASE:", EV["stat_rank_cv_mean_MASE"])"""),
    md("## 1. Overall leaderboards (CV folds = model selection; holdout = final check)"),
    code("""cols = ["model", "median_WAPE", "pooled_WAPE", "mean_MASE", "median_MASE", "median_Bias"]
display(pd.read_csv("reports/tables/leaderboard_cv_overall.csv")[cols].round(3))
display(pd.read_csv("reports/tables/leaderboard_holdout_overall.csv")[cols].round(3))"""),
    md("""**How to read it:** *median WAPE* is the typical series' error as a share of its volume (dominated by the 120 SKUs, where errors of 40-70% are normal for weekly item demand); *pooled WAPE* weights by volume; *mean MASE* is the cross-series ranking metric (< 1 = better than the in-sample seasonal naive).

**Observed (CV):** Ensemble 0.681 < SARIMA 0.701 < DHR-ARIMA 0.708 < LightGBM 0.712 < ETS 0.750 < SES 0.772 < Prophet 0.774 < MovingAverage 0.789 < Naive 0.860 < **SeasonalNaive 1.020**.
**Holdout** keeps the same top group (Ensemble 0.668, LightGBM 0.678, SARIMA 0.699), so the CV ranking was not a fluke of the folds."""),
    md("## 2. By series group: aggregates vs SKUs"),
    code("""g = pd.read_csv("reports/tables/leaderboard_cv_by_group.csv")
display(g.pivot(index="model", columns="group", values=["mean_MASE", "median_WAPE", "pooled_WAPE", "median_Bias"]).round(3).sort_values(("mean_MASE", "Aggregates (91)")))
gh = pd.read_csv("reports/tables/leaderboard_holdout_by_group.csv")
display(gh.pivot(index="model", columns="group", values=["mean_MASE", "pooled_WAPE"]).round(3))"""),
    md("""**Observed:** on the 91 aggregate series **SARIMA is the best single model** (CV mean MASE 0.636, pooled WAPE 5.3%); on the 120 SKUs **LightGBM is the best single model** (0.733). The Ensemble is best or tied-best in both groups. On the holdout, SARIMA (0.561), LightGBM (0.564) and Ensemble (0.537) are close on aggregates, and LightGBM/Ensemble lead on SKUs.

**Seasonal naive is the worst model on aggregates (MASE 1.11, median bias -11.6%).** This is not a bug: with the strong trend (notebook 02), "same week last year" is systematically one year's growth too low. It is a useful baseline exactly because it shows how much of the signal is *trend*, not seasonality."""),
    md("## 3. By hierarchy level and by demand class"),
    code("""show("07_leaderboard_by_level.png")
show("08_mase_by_class_heatmap.png")"""),
    md("""- **Department level** is where SARIMA is most clearly ahead (0.519 vs 0.566 Ensemble, 0.660 DHR).
- **Total / category / store:** DHR-ARIMA and SARIMA are essentially tied (total 0.326 vs 0.331).
- **SKU classes:** LightGBM or the Ensemble win smooth, erratic and intermittent SKUs; **ETS is best on lumpy SKUs** (0.739); all models are within about 0.07 MASE of each other on intermittent SKUs. There is little signal to find there."""),
    md("## 4. Fold-by-fold stability"),
    code("""display(pd.read_csv("reports/tables/fold_by_fold_mean_MASE.csv").set_index("model").round(3))"""),
    md("""The top group (Ensemble, SARIMA, DHR-ARIMA, LightGBM) stays in the top group in every fold; LightGBM is the best *single* model in cv1 (0.669) but the worst of the four in cv4 (0.741), which illustrates why a single split would be fragile."""),
    md("## 5. Are the differences real? Diebold-Mariano and paired bootstrap"),
    code("""display(pd.read_csv("reports/tables/bootstrap_mase_diff.csv").round(4))
display(pd.read_csv("reports/tables/dm_tests_summary.csv"))"""),
    md("""**Bootstrap across series (mean MASE difference A − B, 95% CI):**
- SARIMA vs LightGBM, aggregates: **−0.050 [−0.084, −0.017]**, so SARIMA is significantly better.
- SARIMA vs LightGBM, SKUs: **+0.018 [−0.004, +0.040]** (Wilcoxon p = 0.036). LightGBM is better, but the CI includes 0: weak evidence.
- SARIMA vs DHR-ARIMA: −0.007 [−0.023, +0.010], a **tie**.
- SARIMA vs ETS, aggregates: −0.089 [−0.121, −0.060], so SARIMA is clearly better; on SKUs the two tie.
- Ensemble vs SARIMA, all series: −0.020 [−0.030, −0.011]. A small but consistent gain (Ensemble better on 68% of series).

**Per-series DM tests** (65 forecast weeks per series): for SARIMA vs LightGBM, **75 of 91 aggregates and 99 of 120 SKUs show no significant difference**. Individual series are too short to separate good models; the differences only become visible when pooled across many series. Versus seasonal naive, SARIMA is significantly better on 43 aggregates and 41 SKUs, and significantly worse on 4 series in total."""),
    md("## 6. Prediction-interval calibration"),
    code("""show("10_interval_coverage.png")
display(pd.read_csv("reports/tables/interval_coverage.csv").round(3))"""),
    md("""**Observed:** SARIMA (CV: 83% / 94% aggregates, 86% / 95% SKUs), DHR-ARIMA and LightGBM's empirical intervals are close to nominal 80% / 95%. **ETS intervals are far too narrow on aggregates (50% / 54% coverage)** and **Prophet's are too narrow too (58% / 78%)**. ETS on SKUs over-covers (96%).
**Why it matters:** an 80% interval that covers 50% would set safety stocks far too low. Calibration has to be *checked*, not assumed. (Likely causes: ETS with heuristic seasonal initialisation understates state uncertainty on trending aggregates; Prophet's intervals ignore seasonal-parameter uncertainty and short-term autocorrelation.)"""),
    md("## 7. What LightGBM relies on, and a robustness check of the price assumption"),
    code("""display(pd.read_csv("reports/tables/lgbm_feature_importance.csv").head(12).round(3))
display(pd.read_csv("reports/tables/lgbm_tuning.csv").round(4).head(6))
display(pd.read_csv("reports/tables/lgbm_ablation_no_price.csv"))"""),
    md("""- The top features are the 2-year and 1-year seasonal lags, store identity, recent level/trend, week of year and the price ratio. LightGBM learns seasonality *from lags* rather than from a fixed seasonal structure.
- **Tuning** (2 pre-CV origins only): all 12 grid points scored mean MASE between 0.708 and 0.736, so hyper-parameters matter little; the chosen point is 31 leaves, min 50 per leaf, 600 trees, learning rate 0.03.
- **Price ablation:** the price features use the *planned* price of the target week (assumed known). Without them, CV MASE moves from 0.733 to 0.741 (SKU) and from 0.686 to 0.692 (aggregates), so **no conclusion depends on that assumption**.
- **Why trees cannot extrapolate:** a tree predicts averages of training targets in its leaves, so it cannot output a value beyond what it has seen. Scaling every target by the series' recent 52-week mean (and predicting the *ratio*) is what lets one model serve series of very different sizes and moving levels; the remaining weakness is visible on strongly trending aggregates, where LightGBM loses to SARIMA."""),
]

# ------------------------------------------------------------------------------------------------ 05
NB05 = [
    md("""# 05 · Why does a model work here and not there?

We relate each model's per-series accuracy (CV folds) to the series characteristics computed on pre-holdout data (`reports/series_characteristics.csv`). Every statement below cites a number produced in this notebook."""),
    code(SETUP),
    code("""import contextlib, io
from src.analysis import why
with contextlib.redirect_stdout(io.StringIO()):
    why.main()
W = J("reports/why_summary.json")
print("regenerated")"""),
    md("## 1. Where SARIMA wins and loses"),
    code("""show("11_sarima_relative_error_vs_characteristics.png")
b = pd.read_csv("reports/tables/why_binned_win_rates.csv")
display(b[["feature", "bin", "n", "sarima_beats_snaive", "sarima_beats_lgbm", "sarima_beats_ets", "median_mase_SARIMA", "median_mase_LightGBM", "median_mase_ETS", "median_mase_SeasonalNaive"]].round(3))"""),
    md("""**Evidence**
- **Seasonal strength:** SARIMA's win rate vs LightGBM rises from **38%** (F_S ≤ 0.2) to 44%, 46% and **66%** (F_S > 0.6); vs ETS from 50% to **85%**. With strong seasonality SARIMA's median MASE is 0.70 vs LightGBM 0.81 and ETS 0.91.
- **Zeros:** on zero-free series SARIMA beats LightGBM on **59%**; on series with zeros only **32-48%**.
- **Trend strength > 0.8:** SARIMA beats LightGBM 59% and ETS 82%.
- **Versus seasonal naive** SARIMA wins 79-95% of series in *every* bin: the lag-52 model plus differencing captures growth that a pure seasonal copy cannot.

**So SARIMA works where** the series is an aggregate (no zeros), has strong trend and seasonality, and has a long, mostly stable history: department and category-type series, where it has the best mean MASE (department 0.519).
**It loses where** demand is SKU-level, zero-inflated and weakly seasonal (median F_S of SKUs about 0.2). ARIMA assumes continuous, roughly Gaussian errors, and with 24-37% zero weeks there is little linear autocorrelation structure to exploit. AutoARIMA itself recognises this: it selects seasonal differencing (D = 1) on only **1.7%** of SKU fits vs 37% of aggregate fits."""),
    code("""display(pd.read_csv("reports/tables/why_by_class.csv").round(3))
print(json.dumps(W["sarima_orders"], indent=1))"""),
    md("## 2. SARIMA vs LightGBM head to head"),
    code("""show("12_sarima_vs_lightgbm.png")
h = pd.read_csv("reports/tables/why_sarima_vs_lgbm_spearman.csv")
display(h[h.p < 0.1].round(3))"""),
    md("""**LightGBM wins** on SKUs (bootstrap ΔMASE +0.018 in its favour). Its advantages there: it **pools information across 211 series** (a single SKU has too little signal, but similar SKUs share patterns), uses **exogenous drivers** (SNAP schedule, events, price), and handles nonlinear effects and zero-inflation without distributional assumptions.
**LightGBM loses** on aggregates with strong trend (bootstrap ΔMASE −0.050 in SARIMA's favour, significant): a tree cannot extrapolate a level beyond its training range, so it relies on the recent-scale normalisation, and it has only 91 smooth aggregate series to learn their specific dynamics from, while SARIMA models each one's own autocorrelation."""),
    md("## 3. ETS, Prophet, DHR and baselines"),
    code("""show("13_winner_share_by_class.png")
display(pd.read_csv("reports/tables/why_winner_counts_cv.csv").set_index(pd.read_csv("reports/tables/why_winner_counts_cv.csv").columns[0]))"""),
    md("""- **DHR-ARIMA** (Fourier + holidays + SNAP) ties SARIMA overall (bootstrap CI includes 0) and wins most total/store series (total 0.326 vs SARIMA 0.331; store 0.431 vs 0.466). Explicit, known drivers help where the monthly SNAP cycle and moving holidays matter most: large, FOODS-heavy aggregates.
- **ETS** is clearly worse than SARIMA on aggregates (ΔMASE −0.089 in SARIMA's favour), and its intervals are badly under-covered there. With s = 52 it must carry 52 seasonal states initialised heuristically. It does best on **lumpy SKUs** (0.739, best of all), where heavy smoothing of a noisy level is what matters.
- **Prophet** is mid-table (0.774) and has the most negative SKU bias (-12.9% median). Its smooth trend + Fourier structure ignores short-term autocorrelation, which matters at weekly grain.
- **Naive / moving average** win a handful of intermittent and lumpy SKUs (Naive wins 8 of 30 intermittent): when demand is mostly noise, the simplest level estimate is hard to beat.
- **Seasonal naive** wins almost nowhere because growth dominates (median aggregate bias −11.6%)."""),
    md("## 4. Where the model choice does not matter"),
    code("""print("CV->holdout winner agreement per series:", round(W["winner_stability_cv_to_holdout"], 3))
print("decision tree accuracy predicting the per-series winner:", round(W["decision_tree_cv_accuracy"], 3), "vs majority baseline", round(W["decision_tree_majority_baseline"], 3))
print(open("reports/tables/why_decision_tree.txt").read()[:1500])"""),
    md("""**This is the most important honest finding:** the model with the best CV score on a series is also best on the holdout for only **12.8%** of series, and a decision tree on the characteristics predicts the per-series winner with 20.4% accuracy vs 18.5% for always guessing the most common winner. Per-series DM tests find no significant difference for most series (notebook 04).
So **per-series model selection is mostly chasing noise**. The robust, reproducible differences are at the **group level** (aggregates vs SKUs, strong vs weak seasonality), and that is how the decision guide below is written.

## 5. Decision guide (from this study's evidence)
| Series characteristics | Prefer | Why (evidence) |
|---|---|---|
| Aggregate (store/department/category), no zeros, F_T > 0.6, F_S > 0.5 | **SARIMA** (or DHR-ARIMA) | best single model on aggregates (MASE 0.636); beats LightGBM 59-66% in these bins; calibrated intervals |
| Large aggregate with known calendar drivers (SNAP, moving holidays) | **DHR-ARIMA** | best on total/category/store; handles the 52.18-week year and the monthly cycle |
| Many related SKU series, zeros 5-40%, weak seasonality | **global LightGBM** | best single model on SKUs (0.733); pools across series; still best without the price assumption (0.741) |
| Lumpy SKU (ADI ≥ 1.32, CV² ≥ 0.49) | **ETS / SES** or the ensemble | ETS best on lumpy (0.739); smoothing beats structure |
| Any level, if you can only deploy one thing | **Ensemble (SARIMA + LightGBM)** | best overall CV (0.681) and holdout (0.668); better than SARIMA on 68% of series |
| Strong growth / trend | never seasonal naive alone | MASE 1.11 on aggregates; −11.6% bias |"""),
]

# ------------------------------------------------------------------------------------------------ 06
NB06 = [
    md("""# 06 · Final forecast and inventory decisions

**What:** refit every candidate on all 277 weeks; forecast 13 weeks ahead (2016-05-21 to 2016-08-13) with 80/95% intervals; convert forecasts into safety stock, reorder points and order-up-to levels; back-test the replenishment policies in a simulation.
**Why refit on everything:** the holdout has done its job (an honest accuracy estimate). Leaving out the latest 13 weeks at deployment would ignore the most relevant information. The **model choice** is not changed: it was fixed on CV."""),
    code(SETUP),
    md("## 1. Forward forecast"),
    code("""print(J("reports/final_forecast_summary.json"))
show("14_final_forecast.png")
display(pd.read_csv("reports/tables/final_forecast_headline.csv").round(1))"""),
    md("""**Model per series = best mean CV MASE at its level**: DHR-ARIMA for total, category and store; SARIMA for departments; Ensemble for store x department and SKUs.

**What the forecast says:** total demand of about **4.27 M units** over the next 13 weeks, **+19.5% vs the same 13 weeks of 2015**. That is consistent with the last 13 actual weeks, which were already +16.2% year on year, and with SARIMA, ETS and the Ensemble giving similar levels. FOODS drives the growth (+22.9%); HOUSEHOLD +9.6%, HOBBIES +7.6%. Weekly peaks fall in the SNAP-heavy first weeks of each month and in early August.
**What it does NOT say:** it knows nothing about future promotions, competitor actions, supply disruptions or new-store openings; prices are assumed unchanged; and the intervals reflect only the variability seen in the past. The CV evidence says that, for the total, a 13-week WAPE of about 2-4% is typical, and that the nominal 80% intervals are roughly calibrated at aggregate level for SARIMA/DHR."""),
    md("""## 2. Inventory policy
- **Safety stock** SS = z · σ · √(L + R) and **order-up-to** S = forecast over L + R weeks + SS (periodic review, R = 1 week); **reorder point** ROP = forecast over L + z · σ · √L.
- **σ is the standard deviation of the model's own forecast errors** (RMSE of CV errors at horizons 1-4), **not** of demand: the forecast already explains trend and seasonality, so only the unexplained part needs buffering.
- **z = Φ⁻¹(service level)**: 1.645 for 95%.
- Lead time is an **assumption** (M5 has none): sensitivity for 1-4 weeks and service levels 90/95/98%. Costs: holding 1 per unit-week, shortage 9 per lost unit (assumed)."""),
    code("""s = pd.read_csv("reports/tables/inventory_simulation_summary.csv")
d = s[(s.lead_time == 2) & (s.service_level == 0.95)]
display(d[["group", "policy", "fill_rate", "stockout_rate", "avg_on_hand", "safety_stock", "total_cost", "total_cost_vs_SeasonalNaive_pct", "avg_on_hand_vs_SeasonalNaive_pct"]].round(3))
show("15_inventory_sensitivity.png")"""),
    md("""**SIMULATED, under stated assumptions; these are not real savings.** At L = 2 weeks and 95% service:
- **SKUs (120):** the Ensemble-driven policy has a cost proxy **42.8% lower** than the seasonal-naive policy (fill rate 93.6% vs 89.5%, average stock 44% lower). LightGBM −42.5%, SARIMA −38.9%. Naive reaches the highest fill (95.0%) but holds 41% more stock than the Ensemble policy.
- **Store x department (70):** Ensemble −55.4%, SARIMA −55.8%, LightGBM −46.4% vs seasonal naive, whose 11% under-forecast during growth causes a 24.9% weekly stock-out rate.
- The ranking is **identical across all 12 lead-time x service-level combinations** (SKU savings 37-43%, store x department 43-66%).

**Honest caveats:** (1) much of the gain vs seasonal naive comes from that baseline's growth bias in a +16% YoY quarter; (2) at SKU level the policies under-deliver the 95% target (weekly stock-out rate 6.1-7.9% vs 5%), because σ comes from only 16 CV errors per series, the √(L+R) rule ignores error autocorrelation, and censored zero-runs understate demand variability; (3) 13 weeks is a short window."""),
    code("""ds_ = s[(s.lead_time == 2) & (s.service_level == 0.95) & s.policy.str.contains("demand-sigma|Ensemble")]
display(ds_[["group", "policy", "safety_stock", "avg_on_hand", "fill_rate", "total_cost"]].round(1))"""),
    md("""**Why σ must come from forecast error:** with the same Ensemble forecasts, sizing safety stock on raw demand variability (`+ demand-sigma`) raises total safety stock by 34% at store x department (92k vs 69k units) and average on-hand by 27%. Fill improves marginally (99.8% vs 99.4%), but the total cost proxy is 12% higher. Seasonality and trend are predictable, so buffering against them is wasted stock."""),
    md("## 3. At-risk list (replenishment / merchandising actions)"),
    code("""r = pd.read_csv("outputs/inventory_recommendations.csv")
print(r.risk.value_counts())
for k in ["STOCK-OUT RISK", "OVERSTOCK RISK"]:
    display(Markdown(f"**{k}** (top 8)"))
    display(r[r.risk == k].sort_values("risk_score", ascending=False).head(8)[["unique_id", "sbc_class", "cv_bias", "sim_fill_rate", "sim_weeks_of_cover", "safety_stock", "reorder_point", "order_up_to_level"]].round(2))"""),
    md("""**Business reading**
- **Stock-out risk** (simulated fill < 90% or CV bias < −10%): mostly lumpy/erratic SKUs whose forecasts run low (e.g. a lumpy WI_2 FOODS SKU with −39% CV bias and 76% simulated fill). *Action:* raise the safety factor or move to a demand-sensing / min-max rule for these items, check for unrecorded stock-outs (censoring), and review the planogram presentation minimum.
- **Overstock risk** (cover > 3 weeks or CV bias > +10%): mostly slow intermittent SKUs where the order-up-to logic holds 10-18 weeks of cover because a single unit is a large share of weekly demand. *Action:* switch these to a fixed minimum presentation quantity / order-multiple policy or to an intermittent-demand method (Croston/TSB), and consider range rationalisation for chronically slow lines.
- In fashion retail (Westside / Zudio type assortments), the same split maps to **core basics** (aggregate-like, forecast with SARIMA/DHR at store x category and allocate down) versus **long-tail SKU-size options** (global ML model + intermittent-demand inventory rules)."""),
]
