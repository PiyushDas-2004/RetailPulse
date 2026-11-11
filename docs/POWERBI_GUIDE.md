# Building the RetailPulse Power BI report (about 45 minutes)

I cannot generate a `.pbix` file from this environment (Power BI Desktop is a Windows GUI application). Everything
it needs is exported as tidy CSVs in `outputs/powerbi/` by `make export`. Follow these steps in Power BI Desktop.

## 1. Load the data (5 min)

*Home -> Get data -> Text/CSV*, then load each file below. In Power Query, check the column types: `ds` = Date,
`*_units`/`forecast_units`/metrics = Decimal number, `h` = Whole number, `recommended`/`is_future` = True/False.

| File | Grain | Role |
|------|-------|------|
| `dim_series.csv` | one row per series (211) | dimension: level, category, department, store, state, demand class, seasonal/trend strength |
| `dim_date.csv` | one row per week | dimension: week start date, year, month, events, SNAP days, `is_future` |
| `dim_model.csv` | one row per model | dimension: model, family (Baseline / Statistical / Machine learning / Combination) |
| `fact_actuals.csv` | series x week | actual weekly units |
| `fact_backtest_forecasts.csv` | series x origin x week x model | back-test forecasts, intervals, actual, error |
| `fact_future_forecasts.csv` | series x future week x model | 13-week forward forecast, intervals, `recommended` flag |
| `fact_metrics.csv` | series x origin x model | WAPE, MASE, bias, plus `sum_ae`/`sum_y` for exact pooled WAPE |
| `fact_inventory_recommendations.csv` | series | safety stock, reorder point, order-up-to level, risk flag |
| `fact_inventory_simulation.csv` | group x policy x lead time x service level | SIMULATED fill rate, stock-outs, cost proxy |

## 2. Relationships (5 min, Model view)

- `dim_series[unique_id]` 1 -> * `fact_actuals`, `fact_backtest_forecasts`, `fact_future_forecasts`, `fact_metrics`, `fact_inventory_recommendations` (`unique_id`)
- `dim_date[ds]` 1 -> * `fact_actuals[ds]`, `fact_backtest_forecasts[ds]`, `fact_future_forecasts[ds]`
- `dim_model[model]` 1 -> * `fact_backtest_forecasts[model]`, `fact_future_forecasts[model]`, `fact_metrics[model]`
- Single-direction filters (dimension -> fact). Mark `dim_date` as the date table (*Table tools -> Mark as date table*).

## 3. DAX measures (10 min)

Create a measures table (*Enter data*, name it `_Measures`) and add:

```DAX
Actual Units = SUM ( fact_actuals[actual_units] )

Forecast Units (future) =
    CALCULATE ( SUM ( fact_future_forecasts[forecast_units] ), fact_future_forecasts[recommended] = TRUE () )

Backtest Forecast = SUM ( fact_backtest_forecasts[forecast_units] )
Backtest Actual   = SUM ( fact_backtest_forecasts[actual_units] )

-- WAPE pooled exactly: sum of absolute errors / sum of actuals (NOT an average of percentages)
WAPE = DIVIDE ( SUM ( fact_backtest_forecasts[abs_error] ), SUM ( fact_backtest_forecasts[actual_units] ) )

-- Bias: + means over-forecast (overstock risk), - means under-forecast (stock-out risk)
Bias % = DIVIDE ( SUM ( fact_backtest_forecasts[error] ), SUM ( fact_backtest_forecasts[actual_units] ) )

Forecast vs Actual % = DIVIDE ( [Backtest Forecast] - [Backtest Actual], [Backtest Actual] )

Mean MASE = AVERAGE ( fact_metrics[MASE] )

WAPE vs Seasonal Naive =
VAR m  = [WAPE]
VAR sn = CALCULATE ( [WAPE], dim_model[model] = "SeasonalNaive" )
RETURN DIVIDE ( m - sn, sn )

80% Interval Coverage =
DIVIDE (
    COUNTROWS ( FILTER ( fact_backtest_forecasts,
        fact_backtest_forecasts[actual_units] >= fact_backtest_forecasts[lo80]
            && fact_backtest_forecasts[actual_units] <= fact_backtest_forecasts[hi80] ) ),
    COUNTROWS ( FILTER ( fact_backtest_forecasts, NOT ISBLANK ( fact_backtest_forecasts[lo80] ) ) ) )

Total Safety Stock = SUM ( fact_inventory_recommendations[safety_stock] )
At-Risk Series = CALCULATE ( COUNTROWS ( fact_inventory_recommendations ), fact_inventory_recommendations[risk] <> "ok" )
```

Use a slicer on `fact_backtest_forecasts[origin]` (cv1..cv4, holdout) so that WAPE is always read for a defined backtest window.

## 4. Suggested pages (25 min)

1. **Executive overview**: cards for holdout WAPE of the recommended model and of SeasonalNaive, and `WAPE vs Seasonal Naive`; a line chart of `Actual Units` + `Forecast Units (future)` by `dim_date[ds]` with `dim_series[level] = "total"`; slicers for level/category/store.
2. **Model leaderboard**: a matrix with `dim_model[model]` on rows and `dim_series[level]` on columns, values = `Mean MASE` (conditional formatting: lower = better); a bar chart of `WAPE` by model with an origin slicer; a `Bias %` bar chart.
3. **Why models win**: a matrix of `Mean MASE` by `dim_model[model]` x `dim_series[sbc_class]` (filter level = sku_store); a scatter of `dim_series[seasonal_strength]` (x) vs `Mean MASE` (y), legend = model.
4. **Forecast explorer**: a series slicer (`dim_series[unique_id]`); a line chart of backtest forecast vs actual (legend = model), plus the future forecast with lo/hi bands (use error bars, or two extra lines for `lo80`/`hi80`).
5. **Inventory**: a table from `fact_inventory_recommendations` (unique_id, risk, safety_stock, reorder_point, order_up_to_level, sim_fill_rate) sorted by risk_score; a line chart from `fact_inventory_simulation` (x = lead_time, y = total_cost, legend = policy, slicers for group and service_level). Add a text box: *"Simulated under assumed lead times and costs; not realised savings."*

## 5. Honesty notes to keep on the report

- Data: public M5 (Walmart, US, 2011-2016; food/household/hobbies). It is not Trent/Westside/Zudio data.
- Inventory numbers are from a simulation with assumed lead times, service levels and unit costs.
- Forecasts are probabilistic estimates; the intervals say how wrong they can plausibly be.
