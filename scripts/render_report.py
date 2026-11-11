"""Render REPORT.md from docs/report_template.md, filling {{table_name}} placeholders straight from the result CSVs,
so no table in the report is hand-typed.   Usage: python -m scripts.render_report
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys

import pandas as pd

from src.utils import ROOT

T = ROOT / "reports/tables"


def md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def piv(name, idx, col, val, order=None, f=3):
    d = pd.read_csv(T / name).pivot(index=idx, columns=col, values=val)
    if order:
        d = d[[c for c in order if c in d.columns]]
    d = d.sort_values(d.columns[0]).reset_index()
    return md(d, f)


def env() -> str:
    pk = {}
    for line in open(ROOT / "requirements.txt"):
        if "==" in line:
            k, v = line.strip().split("==")
            pk[k] = v
    cpu = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
    rows = [("Python", sys.version.split()[0]), ("OS", f"{platform.system()} {platform.release()} ({platform.machine()})"),
            ("CPU cores", cpu)] + sorted(pk.items())
    return md(pd.DataFrame(rows, columns=["component", "version"]))


def tables() -> dict:
    lv = ["total", "category", "department", "store", "store_dept", "sku_store"]
    cls = ["aggregate (all smooth)", "SKU-smooth", "SKU-erratic", "SKU-intermittent", "SKU-lumpy"]
    cols = ["model", "median_WAPE", "pooled_WAPE", "mean_MASE", "median_MASE", "median_Bias", "mean_sMAPE"]
    inv = pd.read_csv(T / "inventory_simulation_summary.csv")
    d = inv[(inv.lead_time == 2) & (inv.service_level == 0.95)]
    sens = inv[inv.policy.isin(["Naive", "SARIMA", "LightGBM", "Ensemble"])].pivot_table(
        index=["group", "service_level", "lead_time"], columns="policy", values="total_cost_vs_SeasonalNaive_pct").reset_index()
    ws = pd.read_csv(T / "why_binned_win_rates.csv")
    rec = pd.read_csv(ROOT / "outputs/inventory_recommendations.csv")
    return {
        "lb_cv": md(pd.read_csv(T / "leaderboard_cv_overall.csv")[cols]),
        "lb_ho": md(pd.read_csv(T / "leaderboard_holdout_overall.csv")[cols]),
        "lb_cv_group_mase": piv("leaderboard_cv_by_group.csv", "model", "group", "mean_MASE"),
        "lb_cv_group_pooled": piv("leaderboard_cv_by_group.csv", "model", "group", "pooled_WAPE"),
        "lb_cv_group_bias": piv("leaderboard_cv_by_group.csv", "model", "group", "median_Bias"),
        "lb_ho_group_mase": piv("leaderboard_holdout_by_group.csv", "model", "group", "mean_MASE"),
        "lb_ho_group_pooled": piv("leaderboard_holdout_by_group.csv", "model", "group", "pooled_WAPE"),
        "lb_cv_level": piv("leaderboard_cv_by_level.csv", "model", "level", "mean_MASE", lv),
        "lb_cv_level_wape": piv("leaderboard_cv_by_level.csv", "model", "level", "median_WAPE", lv),
        "lb_ho_level": piv("leaderboard_holdout_by_level.csv", "model", "level", "mean_MASE", lv),
        "lb_cv_class": piv("leaderboard_cv_by_sku_class.csv", "model", "sku_class", "mean_MASE", cls),
        "lb_ho_class": piv("leaderboard_holdout_by_sku_class.csv", "model", "sku_class", "mean_MASE", cls),
        "lb_cv_seas": piv("leaderboard_cv_by_seasonality_band.csv", "model", "seasonality_band", "mean_MASE"),
        "lb_cv_trend": piv("leaderboard_cv_by_trend_band.csv", "model", "trend_band", "mean_MASE"),
        "fold_mase": md(pd.read_csv(T / "fold_by_fold_mean_MASE.csv")),
        "fold_wape": md(pd.read_csv(T / "fold_by_fold_median_WAPE.csv")),
        "bootstrap": md(pd.read_csv(T / "bootstrap_mase_diff.csv"), 4),
        "dm": md(pd.read_csv(T / "dm_tests_summary.csv")),
        "coverage": md(pd.read_csv(T / "interval_coverage.csv")),
        "why_bins": md(ws[["feature", "bin", "n", "sarima_beats_snaive", "sarima_beats_lgbm", "sarima_beats_ets",
                           "median_mase_SARIMA", "median_mase_LightGBM", "median_mase_ETS", "median_mase_DHR", "median_mase_SeasonalNaive"]]),
        "why_class": md(pd.read_csv(T / "why_by_class.csv").rename(columns={"Unnamed: 0": "class"})),
        "why_winners": md(pd.read_csv(T / "why_winner_counts_cv.csv").rename(columns={"Unnamed: 0": "class"})),
        "sarima_grid_top": md(pd.read_csv(T / "sarima_total_top_aicc_vs_cv.csv"), 4),
        "sarima_lb": md(pd.read_csv(T / "sarima_total_ljung_box.csv"), 4),
        "sarima_transform": md(pd.read_csv(T / "sarima_total_transform_cv.csv").pivot(index="origin", columns="transform", values="MASE").reset_index(), 4),
        "stationarity": md(pd.read_csv(T / "stationarity_tests.csv").groupby(["transform", "verdict"]).size().unstack(fill_value=0).reset_index()),
        "events": md(pd.read_csv(T / "event_effects.csv"), 2),
        "snap": md(pd.read_csv(T / "snap_effect.csv"), 1),
        "lgbm_imp": md(pd.read_csv(T / "lgbm_feature_importance.csv").head(12)[["feature", "gain_share"]]),
        "lgbm_tuning": md(pd.read_csv(T / "lgbm_tuning.csv"), 4),
        "lgbm_ablation": md(pd.read_csv(T / "lgbm_ablation_no_price.csv"), 4),
        "inv_default": md(d[["group", "policy", "fill_rate", "stockout_rate", "avg_on_hand", "safety_stock", "lost_units", "total_cost",
                             "total_cost_vs_SeasonalNaive_pct", "avg_on_hand_vs_SeasonalNaive_pct"]], 3),
        "inv_sens": md(sens, 1),
        "risk_counts": md(rec.assign(level=rec.unique_id.str.startswith("SKU").map({True: "SKU", False: "store_dept"}))
                          .groupby(["level", "risk"]).size().rename("n").reset_index()),
        "risk_stock": md(rec[rec.risk == "STOCK-OUT RISK"].sort_values("risk_score", ascending=False).head(8)[
            ["unique_id", "sbc_class", "cv_bias", "sim_fill_rate", "safety_stock", "reorder_point", "order_up_to_level"]], 2),
        "risk_over": md(rec[rec.risk == "OVERSTOCK RISK"].sort_values("risk_score", ascending=False).head(8)[
            ["unique_id", "sbc_class", "cv_bias", "sim_weeks_of_cover", "safety_stock", "reorder_point", "order_up_to_level"]], 2),
        "headline": md(pd.read_csv(T / "final_forecast_headline.csv"), 1),
        "env": env(),
    }


def main() -> None:
    tpl = open(ROOT / "docs/report_template.md").read()
    tb = tables()
    missing = set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(tb)
    assert not missing, f"unknown placeholders: {missing}"
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: tb[m.group(1)], tpl)
    open(ROOT / "REPORT.md", "w").write(out)
    print("REPORT.md written:", len(out.splitlines()), "lines")


if __name__ == "__main__":
    main()
