# Resume bullets (every number traceable to a table produced by this repo)

1. **Built a leakage-free demand-forecasting benchmark on 211 weekly Walmart M5 series** (91 hierarchy aggregates +
   120 SKU-store series stratified by demand class), comparing 10 models under 4-fold rolling-origin CV and a
   13-week holdout. A SARIMA + LightGBM ensemble reached mean MASE **0.681 vs 1.020** for seasonal naive (holdout
   **0.668 vs 1.051**).
   *Source: `reports/tables/leaderboard_cv_overall.csv`, `leaderboard_holdout_overall.csv` (REPORT §9.1).*

2. **Showed with significance tests where each model works:** SARIMA was the best single model on aggregate series
   (median WAPE **7.9% vs 14.9%** for seasonal naive, a 47% reduction; significantly better than LightGBM, bootstrap
   ΔMASE −0.050), while a global LightGBM was best on SKU demand (median WAPE **70.3% vs 93.4%**). Validated with
   Diebold-Mariano tests, a paired bootstrap and interval-coverage checks.
   *Source: `reports/tables/leaderboard_cv_by_group.csv`, `bootstrap_mase_diff.csv` (REPORT §9.2, §9.5).*

3. **Turned forecasts into replenishment decisions** (safety stock from CV forecast error, reorder point,
   order-up-to level). In a simulated 13-week lost-sales back-test (assumed 2-week lead time, 95% service), the
   ensemble-driven policy cut the holding + shortage cost proxy by **42.8%** and raised fill rate from **89.5% to
   93.6%** vs a seasonal-naive policy across 120 SKUs. This is a simulated result under stated assumptions.
   *Source: `reports/tables/inventory_simulation_summary.csv` (REPORT §12.2).*
