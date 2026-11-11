"""Generate the six project notebooks (markdown WHY cells + code). Execute them with `make notebooks`.

Keeping notebook sources in one script makes them reviewable in diffs; the executed .ipynb files (with outputs)
are committed so they render on GitHub.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import os, sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
def show(fig): display(Image(filename=f"reports/figures/{fig}"))
def J(path): return json.load(open(path))"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, ROOT / "notebooks" / name)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------ 01
NB01 = [
    md("""# 01 · Data understanding

**Goal:** know exactly what the data is before modelling it.

| | |
|---|---|
| **What** | Download M5 (Walmart unit sales, 2011-01-29 to 2016-05-22), validate it, and build 211 weekly series. |
| **Why** | Every later number depends on the data being correct and correctly aggregated. A silent error here (for example treating "not yet launched" as "zero demand") would bias every model. |
| **Alternatives** | UCI Online Retail II (one UK web shop, about 2 years, needs heavy transaction cleaning), Rossmann/Favorita (need a Kaggle login, which was not available). See `DECISIONS.md` D-03. |
| **Result** | 30,490 item x store series, 1,941 days, 0 missing cells, 0 duplicates, 0 negatives. |
| **Meaning** | The data is clean at the cell level; the real modelling challenges are *zeros*, *launch dates* and *seasonality*, not dirty records. |"""),
    code(SETUP),
    md("""## 1. Raw files

`make data` runs `src/data/download.py` (HTTP, with a sparse git clone as fallback, SHA-256 verified) and then `src/data/prepare.py`.
Raw CSVs live in `data/raw/m5/` and are **git-ignored** (about 490 MB)."""),
    code("""raw = ROOT / "data/raw/m5"
cal = pd.read_csv(raw / "calendar.csv")
prices = pd.read_csv(raw / "sell_prices.csv", nrows=5)
sales_head = pd.read_csv(raw / "sales_train_evaluation.csv", nrows=3, usecols=range(10))
display(cal.head(3)); display(prices); display(sales_head)"""),
    md("""**Column meanings**
- `sales_train_evaluation.csv`: one row per item x store; `d_1 … d_1941` = units sold per day; ids give `item_id`, `dept_id` (7), `cat_id` (FOODS / HOBBIES / HOUSEHOLD), `store_id` (10), `state_id` (CA / TX / WI).
- `calendar.csv`: maps `d_*` to dates; `wm_yr_wk` = Walmart fiscal week (Saturday to Friday); `event_name/type` (sporting, cultural, national, religious); `snap_CA/TX/WI` = days on which SNAP food-stamp benefits are paid.
- `sell_prices.csv`: weekly shelf price per store x item x week. A row exists **only when the item is on sale in that store**."""),
    md("## 2. Validation profile (`reports/data_profile.json`)"),
    code("""prof = J("reports/data_profile.json")
for k in ["files", "date_range_daily", "missing_values", "duplicates", "negative_values", "zero_inflation", "calendar_gaps_days"]:
    print(f"{k:22s}", prof[k])
print("daily total minimum:", prof["daily_total_units"]["min"], "on", prof["daily_total_units"]["min_date"])
print("days < 10% of median:", prof["daily_total_units"]["days_below_10pct_of_median"])"""),
    md("""**Reading it**
- No missing values, duplicates, negative sales or calendar gaps, so no imputation is needed.
- 68% of *daily* item x store cells are zero. At this grain most demand is intermittent, which is one reason to aggregate to weeks.
- The five near-zero days are all **25 December**: stores are closed. These are *real* calendar effects, not data errors, so we keep them (the Christmas week is about 13% below its neighbours, see notebook 02)."""),
    md("""## 3. "No sales" vs "no data": the listing mask

**What:** an item x store is considered *listed* in a week only if `sell_prices.csv` has a price for it.
**Why:** before launch, a zero means "the product did not exist", not "nobody wanted it". Feeding launch zeros to a model would teach it that demand was zero.
**Check:** if the mask is right, *no units* should ever be sold in an unlisted week."""),
    code("""print(json.dumps(prof["listing"], indent=1))"""),
    md("""**Result:** 0 units were sold in unlisted weeks, so the mask is exact. 10,932 item x stores are listed from week 0; the median launch is week 22; there are no relistings after a delisting, so SKU series simply start at their first listed week.
Zeros *while listed* (23.9% of listed item-store weeks) are either genuine zero demand **or unobserved stock-outs**. The data cannot tell these apart (censoring, see notebook 02)."""),
    md("""## 4. Weekly series used for modelling

**Granularity = weekly** (Walmart weeks, Saturday to Friday): 277 complete weeks; the last 2 days are dropped so the final week is not a fake drop.
**Series set** (DECISIONS D-10): all 91 aggregates (total, 3 categories, 7 departments, 10 stores, 70 store x department) plus 120 SKU x store series stratified 30 per Syntetos-Boylan class, split by volume tercile."""),
    code("""meta = pd.read_csv("data/processed/series_meta.csv")
display(meta.level.value_counts().rename("n_series").to_frame())
print(json.dumps(prof["sku_sampling"], indent=1))
display(meta[meta.level == "sku_store"].sample_stratum.value_counts().sort_index().to_frame("n"))"""),
    md("""**Why stratify the SKU sample?** A random draw from the 20,366 eligible SKUs would be about 51% smooth and contain few erratic items. Equal class sizes let us compare models *within* each demand type, which is the core of notebook 05.
**Eligibility (listed at least 200 of 277 weeks):** guarantees at least 2 seasonal cycles before the first forecast origin. The cost is that new-product cold starts are excluded (a stated limitation).

## 5. Limitations of this dataset for real retail demand
1. **No apparel.** Walmart FOODS / HOUSEHOLD / HOBBIES; fashion has shorter life-cycles, size/colour variants and markdown seasons.
2. **Sales ≠ demand.** Stock-outs are not recorded, so observed sales understate demand (censoring).
3. **No promotion flags.** Prices change rarely (1.4% of SKU-weeks) and there is no promo calendar.
4. **No lead times, costs or inventory positions.** The inventory part is a simulation under stated assumptions.
5. **US-specific calendar** (SNAP, Super Bowl, Thanksgiving). An Indian retailer would need Diwali, wedding season, EOSS sales, etc."""),
]

# ------------------------------------------------------------------------------------------------ 02
NB02 = [
    md("""# 02 · Exploratory data analysis

Each finding is followed by **why it matters** for model choice. Everything is regenerated by `src/analysis/eda.py` (run in the next cell), so the figures and numbers below come from code executed in this notebook."""),
    code(SETUP),
    code("""import contextlib, io
from src.analysis import eda
with contextlib.redirect_stdout(io.StringIO()):
    eda.main()
E = J("reports/eda_summary.json")
print("EDA regenerated")"""),
    md("## 1. Level, trend, structural change"),
    code("""show("01_total_and_categories.png")
display(pd.read_csv("reports/tables/total_by_year.csv").round(0))"""),
    md("""**Observed:** the average weekly total grows from about 184k units (2011) to 231k (2012) and 252k (2013), plateaus in 2014 (251k), then rises to 265k (2015) and 283k in early 2016 (pre-holdout weeks; the yearly table includes the holdout). FOODS is about 70% of units. The 2011-2012 rise is steep and not repeated.
**Why it matters:** there is a real **trend with changing slope**. Models that cannot follow the level (seasonal naive repeats last year's level exactly) will under-forecast in growth years. Differencing (d = 1) or a trend component is needed."""),
    md("## 2. STL decomposition and outliers"),
    code("""show("02_stl_total.png")
o = pd.read_csv("reports/tables/outliers_total.csv")
print(len(o), "weeks with |robust z| > 3.5 on the STL remainder;", (pd.to_datetime(o.week_start).dt.year == 2012).sum(), "of them in 2012")
display(o.head(12))"""),
    md("""**Why STL rather than classical decomposition?** STL (LOESS-based) lets the seasonal shape evolve over time and, with `robust=True`, down-weights outliers so they don't distort the trend or seasonal estimates. Classical decomposition assumes a fixed seasonal pattern and loses half a period at each end.

**Observed:** the "outliers" are not isolated spikes. They come in long runs (for example May to September 2012 are all positive). That is the signature of a **level shift / changing seasonal amplitude** that the deliberately smooth STL trend (about 100-week window) does not follow, not of bad data.

**Decision:** **leave** them. Capping would erase a real demand change; deleting would create gaps. Models with a stochastic level (ARIMA with d = 1, ETS) adapt to such shifts; this is noted as a threat to fixed-pattern models."""),
    md("## 3. Calendar effects: day of week, month, week of year, events, SNAP, price"),
    code("""show("03_calendar_effects.png")
display(pd.DataFrame(E["event_effects"]))
display(pd.DataFrame(E["snap_effect"]).round(2))
print("price effect on SKU sample:", {k: round(v, 3) for k, v in E["price_effect"].items()})"""),
    md("""**Observed and why it matters**
- **Day of week:** Saturday and Sunday are about 20% above the average day and Tuesday to Thursday about 12% below. Weekly buckets absorb this completely, which is a big reason to model weekly.
- **Month / week of year:** a modest annual pattern (August is about 4% above average, November about 4% below) and a sharp **Christmas-week dip (-13% vs adjacent weeks)** because stores close on 25 December. Thanksgiving and Easter weeks are about -2%; Super Bowl week is about +4%. So seasonality is **real but moderate** (seasonal strength 0.75 on the total, notebook 03).
- **SNAP:** FOODS sell 10% (CA), 15% (TX) and **29% (WI)** more on SNAP days. SNAP days fall in the first half of each month, which creates a **~4.3-week cycle in weekly data** that a 52-week seasonal model cannot represent. Known-in-advance SNAP regressors (DHR-ARIMAX, Prophet, LightGBM) can.
- **Price:** only 1.4% of SKU-weeks have a price change (median size 10.6%); the log-units vs log-price correlation is -0.03. Price is a **weak driver here**, and there are no promotion flags (a limitation)."""),
    md("## 4. Autocorrelation and differencing (TOTAL)"),
    code("""show("04_acf_pacf_total.png")
display(pd.DataFrame(E["acf_total"]).round(2))"""),
    md("""**Reading the ACF/PACF in words**
- *Level:* the ACF decays slowly (0.85 at lag 1, still 0.71 at lag 13), which is the textbook sign of a **unit root / trend**. Differencing is needed (d = 1).
- *After d = 1:* the lag-1 ACF is about -0.07, but **lag 2 = -0.59, lag 4 = +0.65, lag 13 = +0.66, lag 26 = +0.67 and lag 52 = +0.71**. The alternating lag-2/lag-4 pattern is the ~4-week (SNAP/payday) cycle; the large lag-52 spike is the annual cycle. Both remain, so differencing alone is not enough.
- *After D = 1 (s = 52):* lag-52 ACF ≈ 0.01, so seasonal differencing removes the annual pattern, but lag-1 ACF = 0.90 shows a remaining trend, so d = 1 is also needed.
- *After d = 1 and D = 1:* small negative spikes at lag 1 (-0.20) and lag 52 (-0.10). Small negative lag-1/lag-s autocorrelations after differencing are the classic signature of an **MA(1) x seasonal MA(1)** ("airline") structure, and a hint that we might be close to over-differencing (strong negative lag-1 would be the alarm sign; -0.20 is mild)."""),
    md("## 5. Stationarity tests (21 aggregate series)"),
    code("""st = pd.read_csv("reports/tables/stationarity_tests.csv")
display(st.groupby(["transform", "verdict"]).size().unstack(fill_value=0))
display(st[st.unique_id == "TOTAL"].round(4))"""),
    md("""**How to read ADF and KPSS together:** ADF's null hypothesis is *"unit root (non-stationary)"*; KPSS's null is *"stationary"*. They are complementary:
both say stationary means stationary; both say non-stationary means difference it; disagreement usually means trend-stationarity, a structural break, or low power.

**Observed:** at level, all 21 aggregates are non-stationary or conflicting. After **d = 1, all 21 are stationary by both tests**. Seasonal differencing alone (D = 1) makes only 1 of 21 stationary.
**Implication for SARIMA:** d = 1 is supported by the tests. The tests cannot decide D (they are not seasonal-unit-root tests), so D is chosen with the **seasonal-strength rule** used by `auto.arima`/statsforecast (F_S > 0.64 means D = 1), see notebook 03."""),
    md("## 6. Does variance grow with the level? (Box-Cox question)"),
    code("""print(json.dumps(E["boxcox_total"], indent=1))"""),
    md("""**Observed:** on TOTAL the yearly mean rose by 54% (184k to 283k) while the standard deviation stayed between 16k and 19k. Variance does **not** grow with level, so a log/Box-Cox transform is not needed for the total (the MLE λ of 2.7 just reflects the trend, not variance stabilisation). Across store x department series the log(sd)-on-log(mean) slope has median 0.75, so there is partial level-dependence at lower levels.
**Decision:** no transform by default; tested explicitly for SARIMA on TOTAL in notebook 03 (log changes CV MASE from 0.309 to 0.306, i.e. nothing)."""),
    md("## 7. Series characterisation: the backbone of the 'why' analysis"),
    code("""ch = pd.read_csv("reports/series_characteristics.csv")
show("05_series_characteristics.png")
display(ch.groupby("level")[["trend_strength", "seasonal_strength", "share_zeros", "cv", "adi", "cv2"]].median().round(3))
display(pd.crosstab(ch.level, ch.sbc_class))"""),
    md("""**Definitions**
- *Trend / seasonal strength* (Hyndman): F = max(0, 1 - Var(remainder)/Var(component + remainder)) from the robust STL. 0 means none, 1 means the component explains everything.
- *ADI* = average interval between non-zero weeks; *CV²* = squared coefficient of variation of the non-zero sizes. Syntetos-Boylan cut-offs 1.32 / 0.49 give **smooth, erratic, intermittent, lumpy**.

**Observed:** aggregates are all *smooth* with strong trend (median F_T 0.74-0.92) and moderate-to-strong seasonality (F_S 0.53-0.75). SKUs have weak structure (median F_T 0.14, F_S 0.21), 24% zero weeks and CV about 1.0.
**Why it matters:** this split predicts *before any modelling* that model choice should differ by level. Structure-seeking models (SARIMA/ETS) have something to find in aggregates; at SKU level most of the signal is noise, and the gains over simple methods should be small."""),
    md("## 8. Zeros, stock-outs and censoring"),
    code("""z = pd.read_csv("reports/tables/sku_zero_runs.csv")
print(E["suspected_stockouts"])
show("06_sku_examples.png")"""),
    md("""**Observed:** 59 of 120 sampled SKUs have a run of at least 4 consecutive zero weeks despite normally selling at least 5 units in a non-zero week, which is implausible as random zero demand. These are **suspected stock-outs** (or temporary delistings not captured by the price table).
**Decision:** keep them as observed (we cannot know the true demand), but **state the censoring limitation**: models learn *sales*, which understate *demand* during stock-outs; this biases forecasts and safety stocks downward for these items. With real retailer data, one would use inventory/availability records to flag and impute censored weeks."""),
]

# ------------------------------------------------------------------------------------------------ 03
NB03 = [
    md("""# 03 · Stationarity and SARIMA deep dive (TOTAL series)

**Notation.** SARIMA(p,d,q)(P,D,Q)[s]:
- **p** autoregressive terms: this week depends on the last p weeks;
- **d** ordinary differences: model week-to-week *changes* instead of levels (removes trend);
- **q** moving-average terms: this week depends on the last q *shocks* (forecast errors);
- **P, D, Q** the same three ideas at the **seasonal lag s** (here s = 52: same week last year).

**Protocol.** Orders are *identified* using only weeks 0-211 (the training data of CV fold 1). They are then *evaluated* on the 4 CV folds and the holdout, with parameters re-estimated at each origin. The order choice therefore never sees the data it is scored on."""),
    code(SETUP),
    code("""import contextlib, io
from src.analysis import sarima_deep_dive
with contextlib.redirect_stdout(io.StringIO()):
    sarima_deep_dive.main()
S = J("reports/sarima_deep_dive.json")
print("deep dive regenerated in this notebook")"""),
    md("""## 1. Choosing d and D from tests, not by guessing
- **d:** repeated KPSS tests on the identification window (`ndiffs`), consistent with notebook 02, where all 21 aggregates became stationary after one difference.
- **D:** seasonal-strength test (`nsdiffs`, `seas`): D = 1 if F_S > 0.64.
- **Over-differencing risk:** each unnecessary difference adds a unit root to the MA side (residual lag-1 ACF near -0.5) and inflates forecast variance. We check this in the residual ACF below."""),
    code("""print("ndiffs (KPSS) =", S["ndiffs_kpss"], "| nsdiffs (seasonal strength) =", S["nsdiffs_seas"])
print("STL strengths on identification window:", {k: round(v, 3) for k, v in S["stl_strengths_ident"].items()})
print("=> d, D =", S["chosen_d_D"])"""),
    md("""## 2. Order selection by AICc over a small, justified grid
**Space:** p ∈ {0..3}, q ∈ {0..2}, P ∈ {0,1}, Q ∈ {0,1} (48 models). Higher seasonal orders at s = 52 would need several more years of data to estimate.
**Why AICc:** it trades fit against complexity (with a small-sample correction), so we don't simply pick the most complex model.
**Its limit:** AICc measures *in-sample* fit under model assumptions. It does not guarantee out-of-sample accuracy, so the top candidates are re-checked on CV."""),
    code("""g = pd.read_csv("reports/tables/sarima_total_aicc_grid.csv")
print("grid of", S["grid_size"], "models fitted in", S["grid_seconds"], "s")
display(g.head(8).round(1))
tcv = pd.read_csv("reports/tables/sarima_total_top_aicc_vs_cv.csv")
display(tcv.round(4))"""),
    md("""**Observed:** the AICc winner is **SARIMA(3,1,2)(0,1,1)[52]**, but the top 6 models lie within about 4 AICc units of each other, and their out-of-sample ranks differ from the AICc ranks: the AICc #3, (3,1,0)(0,1,1), has the best CV MASE (0.301 vs 0.309). All six score MASE 0.30-0.32.
**Meaning:** many orders describe this series about equally well. Information criteria can't separate them, and *neither can CV*: the differences are tiny. We keep the AICc winner (pre-registered rule) and note the result is robust to the exact order. This is also why an automatic search (AutoARIMA) is acceptable for the 211-series benchmark."""),
    code("""for k in ["autoarima_approx_True", "autoarima_approx_False"]:
    print(k, S[k])"""),
    md("""The stepwise AutoARIMA (Hyndman-Khandakar) used in the benchmark picks a nearby model, (3,1,1)(1,1,1) with the CSS approximation or (3,1,0)(0,1,0) with exact ML, at 2 s vs 19 s per fit. With 211 series x 5 origins, the approximation (also R's `auto.arima` default for long or high-frequency series) is the practical choice.

## 3. Transformation: log vs none (judged out of sample)"""),
    code("""print(S["transform_cv_mean_MASE"])
display(pd.read_csv("reports/tables/sarima_total_transform_cv.csv").pivot(index="origin", columns="transform", values="MASE").round(4))"""),
    md("""**Decision:** no transform. The log version is only 1% better in mean CV MASE (0.306 vs 0.309) and not consistently better fold by fold. Combined with the constant variance found in notebook 02, there is no evidence that variance stabilisation is needed for the total.

## 4. Residual diagnostics: did the model capture the structure?
If the model has captured all predictable structure, the residuals should be **white noise**: no autocorrelation (Ljung-Box p > 0.05), mean about 0, and ideally roughly normal (which the prediction intervals assume)."""),
    code("""show("09_sarima_residual_diagnostics.png")
print(json.dumps(S["diagnostics"], indent=1))
display(pd.read_csv("reports/tables/sarima_total_ljung_box.csv").round(4))"""),
    md("""**Reading it honestly**
- **Autocorrelation:** Ljung-Box rejects white noise at lags 8 (p = 0.003) and 13 (p = 0.033), but not at lags 26 (p = 0.14) or 52 (p = 0.08). Lag 4 is not testable (degrees of freedom ≤ 0 after 6 estimated parameters). The visible culprit is a lag-2 spike (-0.16). The differenced data had lag-2 = -0.59 and lag-4 = +0.65 (a ~4-week cycle), so a leftover of the **monthly SNAP/payday rhythm**, which a 52-week seasonal model cannot express, is the most plausible explanation. **The model has captured the annual structure but not all short-cycle structure.** This is evidence for adding exogenous calendar regressors (DHR-ARIMAX / LightGBM), not for adding more ARMA terms.
- **No over-differencing alarm:** there is no large negative lag-1 residual autocorrelation.
- **Normality:** skew 0.23, excess kurtosis 0.68, Jarque-Bera p = 0.053, Shapiro p = 0.26. This is approximately normal with slightly heavy tails, so Gaussian intervals are a reasonable approximation.
- The largest residuals fall in Feb 2013/2015/2016 and late Nov 2014 (Super Bowl / Thanksgiving weeks shifting relative to a fixed 52-week lag)."""),
    md("## 5. Out-of-sample accuracy and interval calibration"),
    code("""show("09b_sarima_total_windows.png")
display(pd.DataFrame(S["fold_metrics_chosen"]).round(4))
print("coverage over all 65 forecast weeks:", S["coverage_chosen_all_windows"])"""),
    md("""**Observed:** WAPE is 2.1-3.3% per 13-week window. Bias is slightly negative in the CV folds (-1% to -2%), i.e. a small under-forecast during growth, and about 0 on the holdout.
**Intervals are too wide:** 95% of actual weeks fell inside the nominal 80% interval and 100% inside the 95% interval (65 points). The intervals are **conservative**, probably because σ² absorbs the 2012 level-shift period and the unmodelled monthly cycle. For inventory this would mean too much safety stock, which is why the inventory module sizes safety stock from **CV forecast errors**, not from model-implied intervals.

## 6. The seasonal-period problem (s = 52) and how it was handled
1. **The year is 52.18 weeks, not 52.** A lag-52 seasonal term drifts by about 1 day per year (and by 1 week in 53-week years), so holidays slide relative to "same week last year".
2. **Computation:** the SARIMA state vector grows with s. statsmodels' `SARIMAX(…, seasonal_order=(P,1,Q,52))` did not finish a single fit within 10 minutes in this environment (with other jobs running), while statsforecast's implementation fits in about 1-2 s.
3. **Estimation:** each seasonal parameter is estimated from only about 4 annual repetitions.

**What we did:** (a) kept classical SARIMA with s = 52 as the benchmark model (fast engine, AutoARIMA order search); (b) added **dynamic harmonic regression** (`DHR-ARIMA`): ARIMA errors + Fourier terms with period 52.18 (K chosen by AICc on pre-CV data) + holiday-week dummies + SNAP-day counts. It handles the non-integer period, moving holidays and the monthly SNAP cycle with few parameters. Notebook 04 compares both under the same protocol.
(Monthly granularity was rejected: it would leave about 64 points and blur the Christmas and Thanksgiving weeks.)"""),
]

if __name__ == "__main__":
    (ROOT / "notebooks").mkdir(exist_ok=True)
    write("01_data_understanding.ipynb", NB01)
    write("02_eda.ipynb", NB02)
    write("03_stationarity_and_sarima.ipynb", NB03)
    try:
        from scripts.notebooks_results import NB04, NB05, NB06  # written after the benchmark results exist
        write("04_model_benchmark.ipynb", NB04)
        write("05_why_analysis.ipynb", NB05)
        write("06_forecast_and_inventory.ipynb", NB06)
    except ImportError as exc:
        print("results notebooks not generated yet:", exc)
