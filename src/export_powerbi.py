"""Export tidy star-schema CSVs for Power BI (see docs/POWERBI_GUIDE.md).

Usage:  python -m src.export_powerbi
"""
from __future__ import annotations

import pandas as pd

from src.data.load import load_calendar, load_series
from src.utils import ROOT, load_config, p

OUT = "outputs/powerbi"
FAMILY = {"Naive": "Baseline", "SeasonalNaive": "Baseline", "MovingAverage": "Baseline", "SES": "Baseline",
          "ETS": "Statistical", "SARIMA": "Statistical", "DHR-ARIMA": "Statistical", "Prophet": "Statistical",
          "LightGBM": "Machine learning", "Ensemble": "Combination"}


def main() -> None:
    cfg = load_config()
    df, meta = load_series()
    cal = load_calendar()
    ch = pd.read_csv(ROOT / "reports/series_characteristics.csv")

    dim_series = meta.merge(ch.drop(columns=["level", "cat_id", "dept_id", "store_id", "n_weeks"]), on="unique_id")
    dim_series.to_csv(p(f"{OUT}/dim_series.csv"), index=False)

    fut = pd.read_csv(ROOT / "outputs/forecasts.csv", parse_dates=["ds"])
    dates = pd.DataFrame({"ds": pd.concat([cal.ds, fut.ds]).drop_duplicates().sort_values()})
    dates = dates.merge(cal, on="ds", how="left")
    dates["year"], dates["month"] = dates.ds.dt.year, dates.ds.dt.month
    dates["iso_week"] = dates.ds.dt.isocalendar().week.astype(int)
    dates["is_future"] = dates.ds > df.ds.max()
    dates.to_csv(p(f"{OUT}/dim_date.csv"), index=False)

    pd.DataFrame({"model": list(FAMILY), "family": list(FAMILY.values())}).to_csv(p(f"{OUT}/dim_model.csv"), index=False)
    df.rename(columns={"y": "actual_units"}).to_csv(p(f"{OUT}/fact_actuals.csv"), index=False)

    allf = pd.read_parquet(ROOT / cfg["paths"]["cache_dir"] / "all_forecasts.parquet")
    allf = allf[["unique_id", "origin", "ds", "h", "model", "mean", "lo80", "hi80", "lo95", "hi95", "y"]].rename(
        columns={"mean": "forecast_units", "y": "actual_units"})
    allf["abs_error"] = (allf.forecast_units - allf.actual_units).abs()
    allf["error"] = allf.forecast_units - allf.actual_units
    allf.to_csv(p(f"{OUT}/fact_backtest_forecasts.csv"), index=False)

    fut.rename(columns={"mean": "forecast_units"}).to_csv(p(f"{OUT}/fact_future_forecasts.csv"), index=False)
    m = pd.read_csv(ROOT / "reports/tables/metrics_by_series_origin.csv")
    m[["unique_id", "origin", "model", "n", "sum_y", "sum_ae", "sum_e", "MAE", "RMSE", "WAPE", "MASE", "Bias", "cov80", "cov95"]].to_csv(
        p(f"{OUT}/fact_metrics.csv"), index=False)
    pd.read_csv(ROOT / "outputs/inventory_recommendations.csv").to_csv(p(f"{OUT}/fact_inventory_recommendations.csv"), index=False)
    pd.read_csv(ROOT / "reports/tables/inventory_simulation_summary.csv").to_csv(p(f"{OUT}/fact_inventory_simulation.csv"), index=False)
    for f in sorted((ROOT / OUT).glob("*.csv")):
        print(f.name, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
