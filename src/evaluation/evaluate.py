"""Score every model under the identical rolling-origin protocol and build leaderboards, tests and plots.

Usage:  python -m src.evaluation.evaluate
Inputs : outputs/cache/stat_forecasts.parquet, outputs/cache/lgbm_forecasts.parquet
Outputs: reports/tables/metrics_by_series_origin.csv, leaderboard_*.csv, coverage.csv, dm_tests.csv, bootstrap.csv,
         series_winners.csv, outputs/cache/all_forecasts.parquet, figures 07-10
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.load import load_series, make_origins, week_index
from src.evaluation.metrics import mase_scale
from src.evaluation.stat_tests import dm_test, paired_bootstrap
from src.plotting import INK2, MODEL_COLORS, PALETTE, plt, save, setup
from src.utils import ROOT, load_config, p

TAB = "reports/tables"
FIG = "reports/figures"
STAT_CANDIDATES = ["ETS", "SARIMA", "DHR-ARIMA", "Prophet"]
BASELINES = ["Naive", "SeasonalNaive", "MovingAverage", "SES"]
MODEL_ORDER = ["Naive", "SeasonalNaive", "MovingAverage", "SES", "ETS", "SARIMA", "DHR-ARIMA", "Prophet", "LightGBM", "Ensemble"]


def load_forecasts(cfg) -> pd.DataFrame:
    c = ROOT / cfg["paths"]["cache_dir"]
    st = pd.read_parquet(c / "stat_forecasts.parquet")
    lg = pd.read_parquet(c / "lgbm_forecasts.parquet")
    cols = ["unique_id", "origin", "week", "h", "model", "mean", "lo80", "hi80", "lo95", "hi95", "info"]
    return pd.concat([st[cols], lg[cols]], ignore_index=True)


def attach_actuals(fc: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    a = df.assign(week=week_index(df.ds))[["unique_id", "week", "ds", "y"]]
    return fc.merge(a, on=["unique_id", "week"], how="left")


def series_scales(df: pd.DataFrame, origins) -> pd.DataFrame:
    rows = []
    wk = week_index(df.ds)
    for uid, g in df.assign(week=wk).groupby("unique_id"):
        for o in origins:
            ytr = g.y[g.week <= o.train_end].to_numpy()
            rows.append({"unique_id": uid, "origin": o.name, "scale": mase_scale(ytr, 52)})
    return pd.DataFrame(rows)


def score(fc: pd.DataFrame) -> pd.DataFrame:
    """Per (series, origin, model) metrics. Sums are kept so that metrics can be pooled exactly later."""
    e = fc["mean"] - fc.y
    fc = fc.assign(ae=e.abs(), se=e ** 2, e=e, sm=np.where((fc.y.abs() + fc["mean"].abs()) == 0, 0,
                                                          2 * e.abs() / (fc.y.abs() + fc["mean"].abs()).replace(0, 1)),
                   in80=(fc.y >= fc.lo80) & (fc.y <= fc.hi80), in95=(fc.y >= fc.lo95) & (fc.y <= fc.hi95),
                   has_pi=fc.lo95.notna())
    g = fc.groupby(["unique_id", "origin", "model"])
    m = g.agg(n=("y", "size"), sum_y=("y", "sum"), sum_ae=("ae", "sum"), sum_se=("se", "sum"), sum_e=("e", "sum"),
              smape=("sm", "mean"), cov80=("in80", "mean"), cov95=("in95", "mean"), has_pi=("has_pi", "all"),
              scale=("scale", "first")).reset_index()
    m["MAE"] = m.sum_ae / m.n
    m["RMSE"] = np.sqrt(m.sum_se / m.n)
    m["WAPE"] = m.sum_ae / m.sum_y.replace(0, np.nan)
    m["Bias"] = m.sum_e / m.sum_y.replace(0, np.nan)
    m["MASE"] = m.MAE / m.scale
    m.loc[~m.has_pi, ["cov80", "cov95"]] = np.nan
    return m


def per_series(m: pd.DataFrame, period: list[str]) -> pd.DataFrame:
    """Collapse origins in `period` into one row per (series, model): pooled WAPE/bias, mean MASE."""
    s = m[m.origin.isin(period)].groupby(["unique_id", "model"]).agg(
        sum_y=("sum_y", "sum"), sum_ae=("sum_ae", "sum"), sum_e=("sum_e", "sum"), MASE=("MASE", "mean"),
        RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), sMAPE=("smape", "mean"), cov80=("cov80", "mean"),
        cov95=("cov95", "mean")).reset_index()
    s["WAPE"] = s.sum_ae / s.sum_y.replace(0, np.nan)
    s["Bias"] = s.sum_e / s.sum_y.replace(0, np.nan)
    return s


def leaderboard(s: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    keys = ["model"] + ([by] if by else [])
    lb = s.groupby(keys).agg(n_series=("unique_id", "nunique"), median_WAPE=("WAPE", "median"),
                             pooled_WAPE=("sum_ae", "sum"), sum_y=("sum_y", "sum"), mean_MASE=("MASE", "mean"),
                             median_MASE=("MASE", "median"), median_Bias=("Bias", "median"), mean_sMAPE=("sMAPE", "mean"),
                             mean_RMSE=("RMSE", "mean")).reset_index()
    lb["pooled_WAPE"] = lb.pooled_WAPE / lb.sum_y
    lb = lb.drop(columns="sum_y")
    lb["rank_mean_MASE"] = lb.groupby(by)["mean_MASE"].rank() if by else lb.mean_MASE.rank()
    return lb.sort_values(([by] if by else []) + ["mean_MASE"])


def main() -> None:
    setup()
    cfg = load_config()
    df, meta = load_series()
    ch = pd.read_csv(ROOT / "reports/series_characteristics.csv")
    origins = make_origins(cfg)
    eval_origins = [o for o in origins if o.kind != "tuning"]
    cv = [o.name for o in origins if o.kind == "cv"]

    fc = load_forecasts(cfg)
    fc = fc[fc.origin.isin([o.name for o in eval_origins])]

    # ---- ensemble: components chosen from CV results only (best statistical model by mean CV MASE + LightGBM)
    fc = attach_actuals(fc, df).merge(series_scales(df, eval_origins), on=["unique_id", "origin"])
    m0 = score(fc)
    s0 = per_series(m0, cv)
    stat_rank = s0[s0.model.isin(STAT_CANDIDATES)].groupby("model").MASE.mean().sort_values()
    best_stat = stat_rank.index[0]
    ens = fc[fc.model.isin([best_stat, "LightGBM"])].groupby(["unique_id", "origin", "week", "h"]).agg(
        mean=("mean", "mean"), y=("y", "first"), ds=("ds", "first"), scale=("scale", "first")).reset_index()
    ens["model"] = "Ensemble"
    ens["info"] = f"mean({best_stat}, LightGBM)"
    for c in ["lo80", "hi80", "lo95", "hi95"]:
        ens[c] = np.nan
    fc = pd.concat([fc, ens[fc.columns]], ignore_index=True)
    fc.to_parquet(ROOT / cfg["paths"]["cache_dir"] / "all_forecasts.parquet", index=False)

    m = score(fc)
    m.to_csv(p(f"{TAB}/metrics_by_series_origin.csv"), index=False)
    attrs = ch[["unique_id", "level", "sbc_class", "seasonality_band", "trend_band", "seasonal_strength", "trend_strength",
                "share_zeros", "cv", "adi", "cv2", "mean"]]
    s_cv = per_series(m, cv).merge(attrs, on="unique_id")
    s_ho = per_series(m, ["holdout"]).merge(attrs, on="unique_id")
    s_cv.to_csv(p(f"{TAB}/series_metrics_cv.csv"), index=False)
    s_ho.to_csv(p(f"{TAB}/series_metrics_holdout.csv"), index=False)
    for s in (s_cv, s_ho):
        s["group"] = np.where(s.level == "sku_store", "SKU (120)", "Aggregates (91)")
        s["sku_class"] = np.where(s.level == "sku_store", "SKU-" + s.sbc_class, "aggregate (all smooth)")

    summary = {"ensemble_components": [best_stat, "LightGBM"], "stat_rank_cv_mean_MASE": stat_rank.round(4).to_dict()}
    for name, s in [("cv", s_cv), ("holdout", s_ho)]:
        leaderboard(s).to_csv(p(f"{TAB}/leaderboard_{name}_overall.csv"), index=False)
        for by in ["group", "level", "sku_class", "seasonality_band", "trend_band"]:
            leaderboard(s, by).to_csv(p(f"{TAB}/leaderboard_{name}_by_{by}.csv"), index=False)
    # fold-by-fold
    fold = m.groupby(["origin", "model"]).agg(mean_MASE=("MASE", "mean"), median_WAPE=("WAPE", "median")).reset_index()
    fold.pivot(index="model", columns="origin", values="mean_MASE").reindex(MODEL_ORDER).to_csv(p(f"{TAB}/fold_by_fold_mean_MASE.csv"))
    fold.pivot(index="model", columns="origin", values="median_WAPE").reindex(MODEL_ORDER).to_csv(p(f"{TAB}/fold_by_fold_median_WAPE.csv"))
    m.assign(group=np.where(m.unique_id.str.startswith("SKU_"), "SKU", "Aggregate")).groupby(["group", "origin", "model"]).agg(
        mean_MASE=("MASE", "mean"), median_WAPE=("WAPE", "median")).reset_index().to_csv(p(f"{TAB}/fold_by_fold_by_group.csv"), index=False)

    # ---- winners per series (CV mean MASE)
    win = s_cv.loc[s_cv.groupby("unique_id").MASE.idxmin(), ["unique_id", "model", "MASE", "level", "sbc_class",
                                                              "seasonal_strength", "trend_strength", "share_zeros"]]
    win = win.rename(columns={"model": "cv_winner", "MASE": "cv_winner_MASE"})
    win.to_csv(p(f"{TAB}/series_winners.csv"), index=False)
    summary["winner_counts"] = win.groupby(["level", "cv_winner"]).size().unstack(fill_value=0).to_dict("index")

    # ---- prediction-interval coverage
    cov = []
    for name, period in [("cv", cv), ("holdout", ["holdout"])]:
        x = m[m.origin.isin(period) & m.cov95.notna()].assign(group=lambda d: np.where(d.unique_id.str.startswith("SKU_"), "SKU", "Aggregate"))
        # weight by number of points (all windows are 13 weeks, so the mean of window coverage = point coverage)
        c = x.groupby(["model", "group"]).agg(cov80=("cov80", "mean"), cov95=("cov95", "mean"), n_windows=("cov80", "size")).reset_index()
        c["period"] = name
        cov.append(c)
    cov = pd.concat(cov)
    cov.to_csv(p(f"{TAB}/interval_coverage.csv"), index=False)

    # ---- statistical tests: DM per series + bootstrap across series
    lb_cv = leaderboard(s_cv)
    top = lb_cv.model.tolist()[:3]
    pairs = sorted({tuple(x) for x in [(top[0], top[1]), (top[0], top[2]), (top[1], top[2]), ("SARIMA", "SeasonalNaive"),
                                      ("SARIMA", "LightGBM"), ("SARIMA", "ETS"), ("SARIMA", "DHR-ARIMA"), ("LightGBM", "SeasonalNaive")]
                    if x[0] != x[1]})
    err = fc[fc.origin.isin(cv + ["holdout"])].assign(e=lambda d: d["mean"] - d.y).sort_values(["unique_id", "week"])
    E = err.pivot_table(index=["unique_id", "origin", "week"], columns="model", values="e")
    dm_rows, bs_rows = [], []
    for a, b in pairs:
        for uid, g in E.groupby(level=0):
            stat, pv = dm_test(g[a].to_numpy(), g[b].to_numpy(), h=cfg["problem"]["horizon"])
            dm_rows.append({"model_a": a, "model_b": b, "unique_id": uid, "dm_stat": stat, "p": pv})
        for grp, sub in [("all", s_cv), ("Aggregates", s_cv[s_cv.level != "sku_store"]), ("SKU", s_cv[s_cv.level == "sku_store"])]:
            pa = sub[sub.model == a].set_index("unique_id").MASE
            pb = sub[sub.model == b].set_index("unique_id").MASE.reindex(pa.index)
            bs_rows.append({"model_a": a, "model_b": b, "group": grp, **paired_bootstrap(pa.values, pb.values, seed=cfg["seed"])})
    dm = pd.DataFrame(dm_rows).merge(meta[["unique_id", "level"]], on="unique_id")
    dm.to_csv(p(f"{TAB}/dm_tests_by_series.csv"), index=False)
    dm["group"] = np.where(dm.level == "sku_store", "SKU", "Aggregate")
    dm["outcome"] = np.where(dm.p >= 0.05, "no significant difference", np.where(dm.dm_stat < 0, "A significantly better", "B significantly better"))
    dms = dm.groupby(["model_a", "model_b", "group", "outcome"]).size().unstack(fill_value=0).reset_index()
    dms.to_csv(p(f"{TAB}/dm_tests_summary.csv"), index=False)
    pd.DataFrame(bs_rows).to_csv(p(f"{TAB}/bootstrap_mase_diff.csv"), index=False)
    # DM on the TOTAL series (the one a CFO would ask about)
    summary["dm_total_series"] = dm[dm.unique_id == "TOTAL"][["model_a", "model_b", "dm_stat", "p"]].round(4).to_dict("records")
    summary["top3_cv"] = top

    # ---- figures
    lvl_order = ["total", "category", "department", "store", "store_dept", "sku_store"]
    lbl = leaderboard(s_cv, "level")
    models = [x for x in MODEL_ORDER if x in lbl.model.unique()]
    fig, axes = plt.subplots(1, 6, figsize=(15, 4.4), sharey=True)
    for ax, lv in zip(axes, lvl_order):
        d = lbl[lbl.level == lv].set_index("model").reindex(models)
        ax.barh(range(len(models)), d.mean_MASE, color=[MODEL_COLORS[x] for x in models], height=0.65)
        ax.axvline(1, color=INK2, lw=0.8, ls="--")
        ax.set_title(f"{lv}\n(n={int(d.n_series.max())})", fontsize=9.5)
        ax.set_yticks(range(len(models)), models)
        ax.invert_yaxis()
        best = d.mean_MASE.idxmin()
        ax.text(d.mean_MASE[best], models.index(best), f" {d.mean_MASE[best]:.2f}", va="center", fontsize=8, color=INK2)
    fig.suptitle("Mean MASE by hierarchy level, 4 rolling-origin CV folds (lower is better; < 1 beats in-sample seasonal naive)", fontweight="bold")
    fig.tight_layout()
    save(fig, p(f"{FIG}/07_leaderboard_by_level.png"))

    # heatmap: model x class (relative to seasonal naive)
    piv = leaderboard(s_cv, "sku_class").pivot(index="model", columns="sku_class", values="mean_MASE").reindex(models)
    cols = ["aggregate (all smooth)", "SKU-smooth", "SKU-erratic", "SKU-intermittent", "SKU-lumpy"]
    piv = piv[cols]
    rel = piv / piv.loc["SeasonalNaive"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(rel.values, cmap="RdBu_r", vmin=0.6, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(cols)), [c.replace(" (all smooth)", "") for c in cols], rotation=15)
    ax.set_yticks(range(len(models)), models)
    ax.grid(False)
    for i in range(len(models)):
        for j in range(len(cols)):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(rel.values[i, j] - 1) > 0.28 else "#0b0b0b",
                    fontweight="bold" if v == np.nanmin(piv.values[:, j]) else "normal")
    ax.set_title("Mean CV MASE by demand class (colour = ratio to seasonal naive; bold = best)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="MASE / SeasonalNaive MASE")
    save(fig, p(f"{FIG}/08_mase_by_class_heatmap.png"))

    # coverage chart
    cc = cov[cov.period == "holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, lvl, nom in zip(axes, ["cov80", "cov95"], [0.8, 0.95]):
        for k, grp in enumerate(["Aggregate", "SKU"]):
            d = cc[cc.group == grp].set_index("model").reindex([x for x in models if x in cc.model.unique()])
            ax.barh(np.arange(len(d)) + k * 0.38, d[lvl], height=0.36, color=PALETTE[k], label=grp)
        ax.axvline(nom, color=INK2, ls="--", lw=1)
        ax.set_yticks(np.arange(len(d)) + 0.19, d.index)
        ax.set_title(f"Empirical coverage of nominal {int(nom*100)}% interval (holdout)")
        ax.set_xlim(0, 1.02)
    axes[0].legend(loc="lower left")
    axes[0].invert_yaxis()
    save(fig, p(f"{FIG}/10_interval_coverage.png"))

    json.dump(summary, open(p("reports/evaluation_summary.json"), "w"), indent=1, default=str)
    print(lb_cv.round(3).to_string(index=False))
    print(leaderboard(s_ho).round(3).to_string(index=False))
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
