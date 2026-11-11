"""Exploratory analysis: figures + tables used by REPORT.md and notebook 02/03.

Usage:  python -m src.analysis.eda
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from src.data.load import load_calendar, load_prices, load_series, make_origins
from src.features.characteristics import characteristics_table
from src.plotting import INK2, PALETTE, plt, save, setup
from src.utils import ROOT, load_config, p

warnings.filterwarnings("ignore")
FIG = "reports/figures"
TAB = "reports/tables"


def stationarity_tests(y: np.ndarray) -> dict:
    """ADF (H0: unit root) and KPSS (H0: level-stationary) on y."""
    adf_p = adfuller(y, autolag="AIC")[1]
    kpss_p = kpss(y, regression="c", nlags="auto")[1]
    if adf_p < 0.05 and kpss_p > 0.05:
        verdict = "stationary (both agree)"
    elif adf_p >= 0.05 and kpss_p <= 0.05:
        verdict = "non-stationary (both agree)"
    elif adf_p < 0.05 and kpss_p <= 0.05:
        verdict = "conflict: ADF rejects unit root, KPSS rejects stationarity (trend/difference-stationary?)"
    else:
        verdict = "conflict: neither rejects (low power)"
    return {"adf_p": adf_p, "kpss_p": kpss_p, "verdict": verdict}


def main() -> None:
    setup()
    cfg = load_config()
    df, meta = load_series()
    cal = load_calendar()
    origins = make_origins(cfg)
    hold_start = origins[-1].test_start
    out: dict = {}

    piv = df.pivot(index="ds", columns="unique_id", values="y")
    pre = piv.iloc[:hold_start]

    # ---------------------------------------------------------------- 1. total + categories over time
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    axes[0].plot(piv.index, piv["TOTAL"] / 1e3, color=PALETTE[0])
    axes[0].axvspan(piv.index[hold_start], piv.index[-1], color="#e4e3df", alpha=0.6, lw=0)
    axes[0].text(piv.index[hold_start], axes[0].get_ylim()[1] * 0.97, " holdout", va="top", color=INK2)
    axes[0].set_title("Total weekly unit sales, all 10 stores (thousands)")
    for i, c in enumerate(["CAT_FOODS", "CAT_HOUSEHOLD", "CAT_HOBBIES"]):
        axes[1].plot(piv.index, piv[c] / 1e3, color=PALETTE[i], label=c.replace("CAT_", ""))
        axes[1].text(piv.index[-1], piv[c].iloc[-1] / 1e3, " " + c.replace("CAT_", ""), va="center", color=INK2, fontsize=8)
    axes[1].set_title("By category (thousands of units)")
    axes[1].legend(loc="upper left", ncol=3)
    save(fig, p(f"{FIG}/01_total_and_categories.png"))

    # level shifts / growth per year
    tot = piv["TOTAL"]
    yearly = tot.groupby(tot.index.year).agg(["mean", "std", "count"])
    yearly["cv"] = yearly["std"] / yearly["mean"]
    yearly.to_csv(p(f"{TAB}/total_by_year.csv"))
    out["total_yearly_mean"] = yearly["mean"].round(0).to_dict()

    # ---------------------------------------------------------------- 2. STL decomposition of total
    stl = STL(pre["TOTAL"].to_numpy(), period=52, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    for ax, comp, name in zip(axes, [pre["TOTAL"].to_numpy(), stl.trend, stl.seasonal, stl.resid],
                              ["Observed", "Trend", "Seasonal (period 52)", "Remainder"]):
        ax.plot(pre.index, comp / 1e3, color=PALETTE[0] if name != "Remainder" else INK2, lw=1.3)
        ax.set_ylabel(name, fontsize=8.5)
    axes[0].set_title("Robust STL decomposition of total weekly sales (pre-holdout, thousands of units)")
    save(fig, p(f"{FIG}/02_stl_total.png"))
    # robust z of remainder => outliers
    r = stl.resid
    z = (r - np.median(r)) / (1.4826 * stats.median_abs_deviation(r))
    out_idx = np.where(np.abs(z) > 3.5)[0]
    outl = pd.DataFrame({"week_start": pre.index[out_idx].date, "total_units": pre["TOTAL"].iloc[out_idx].values,
                         "stl_remainder": r[out_idx].round(0), "robust_z": z[out_idx].round(2)})
    wkcal = cal.set_index("ds")
    outl["events_in_week"] = [", ".join(k for k in ["christmas", "thanksgiving", "easter", "superbowl"]
                                        if wkcal.loc[pd.Timestamp(d), k]) for d in outl.week_start]
    outl.to_csv(p(f"{TAB}/outliers_total.csv"), index=False)
    out["outliers_total"] = outl.to_dict("records")

    # ---------------------------------------------------------------- 3. calendar effects (daily raw data)
    raw = ROOT / cfg["paths"]["raw_dir"]
    calendar = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    dcols = [c for c in sales.columns if c.startswith("d_")]
    daily = pd.DataFrame({"date": calendar.date.iloc[: len(dcols)].values})
    for c in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        daily[c] = sales.loc[sales.cat_id == c, dcols].to_numpy().sum(0)
    daily["TOTAL"] = daily[["FOODS", "HOUSEHOLD", "HOBBIES"]].sum(1)
    for st in ["CA", "TX", "WI"]:
        daily[f"FOODS_{st}"] = sales.loc[(sales.cat_id == "FOODS") & (sales.state_id == st), dcols].to_numpy().sum(0)
    daily = daily.merge(calendar[["date", "weekday", "month", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]], on="date")
    open_days = daily[~daily.date.dt.strftime("%m-%d").eq("12-25")]
    dow = open_days.groupby("weekday")["TOTAL"].mean()
    dow = dow.reindex(["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    dow_idx = dow / dow.mean()
    mon = open_days.groupby("month")["TOTAL"].mean()
    mon_idx = mon / mon.mean()
    # week-of-year seasonal index from the STL seasonal component (ratio to level)
    seas = pd.Series(stl.seasonal, index=pre.index)
    woy = (seas / stl.trend).groupby(pre.index.isocalendar().week.values).mean()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].bar(range(7), dow_idx.values, color=PALETTE[0], width=0.65)
    axes[0].set_xticks(range(7), [d[:3] for d in dow_idx.index])
    axes[0].axhline(1, color=INK2, lw=0.8)
    axes[0].set_title("Day-of-week index (daily, total)")
    axes[1].bar(range(1, 13), mon_idx.values, color=PALETTE[0], width=0.65)
    axes[1].axhline(1, color=INK2, lw=0.8)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("Month index (avg daily units)")
    axes[2].plot(woy.index, 100 * woy.values, color=PALETTE[0])
    axes[2].axhline(0, color=INK2, lw=0.8)
    axes[2].set_title("Week-of-year seasonal effect (% of trend)")
    axes[2].set_xlabel("ISO week")
    for ax in axes[:2]:
        ax.set_ylim(0.8, max(1.25, ax.get_ylim()[1]))
    save(fig, p(f"{FIG}/03_calendar_effects.png"))
    out["dow_index"] = dow_idx.round(3).to_dict()
    out["month_index"] = mon_idx.round(3).to_dict()
    out["woy_effect_pct_min_max"] = [float(100 * woy.min()), int(woy.idxmin()), float(100 * woy.max()), int(woy.idxmax())]

    # events: weekly total in event weeks vs same-season baseline (mean of the 2 adjacent weeks)
    w = cal.copy()
    w["TOTAL"] = piv["TOTAL"].values
    w["base"] = (w.TOTAL.shift(1) + w.TOTAL.shift(-1)) / 2
    ev_rows = []
    for e in ["christmas", "thanksgiving", "easter", "superbowl"]:
        sub = w[w[e] == 1].dropna(subset=["base"])
        ev_rows.append({"event": e, "n_weeks": len(sub), "mean_lift_vs_adjacent_weeks_pct": float(100 * (sub.TOTAL / sub.base - 1).mean())})
    sub = w[(w.n_events > 0) & (w[["christmas", "thanksgiving", "easter", "superbowl"]].sum(1) == 0)].dropna(subset=["base"])
    ev_rows.append({"event": "other_events", "n_weeks": len(sub), "mean_lift_vs_adjacent_weeks_pct": float(100 * (sub.TOTAL / sub.base - 1).mean())})
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(p(f"{TAB}/event_effects.csv"), index=False)
    out["event_effects"] = ev.round(2).to_dict("records")

    # SNAP: FOODS daily sales on SNAP vs non-SNAP days, per state (same month, open days)
    snap_rows = []
    for st in ["CA", "TX", "WI"]:
        g = open_days.groupby(f"snap_{st}")[f"FOODS_{st}"].mean()
        snap_rows.append({"state": st, "foods_daily_nonSNAP": g[0], "foods_daily_SNAP": g[1], "lift_pct": 100 * (g[1] / g[0] - 1)})
    snap = pd.DataFrame(snap_rows)
    snap.to_csv(p(f"{TAB}/snap_effect.csv"), index=False)
    out["snap_effect"] = snap.round(2).to_dict("records")

    # price effect on the SKU sample: weekly log-change in units vs log-change in price
    pr = load_prices()
    sk = df[df.unique_id.str.startswith("SKU_")].merge(pr, on=["unique_id", "ds"])
    sk = sk.sort_values(["unique_id", "ds"])
    sk["dlp"] = sk.groupby("unique_id").price.transform(lambda s: np.log(s).diff())
    sk["dly"] = sk.groupby("unique_id").y.transform(lambda s: np.log1p(s).diff())
    ch = sk[sk.dlp.abs() > 0.01].dropna()
    pr_stats = {"n_price_changes": int(len(ch)), "share_sku_weeks_with_price_change": float((sk.dlp.abs() > 0.01).mean()),
                "median_abs_price_change_pct": float(100 * ch.dlp.abs().median()),
                "corr_dlogprice_dlogunits": float(np.corrcoef(ch.dlp, ch.dly)[0, 1]),
                "ols_slope_dlogunits_on_dlogprice": float(np.polyfit(ch.dlp, ch.dly, 1)[0])}
    out["price_effect"] = pr_stats

    # ---------------------------------------------------------------- 4. ACF / PACF of total
    y = pre["TOTAL"].to_numpy()
    yl = np.log(y)
    variants = {"level": y, "first difference (d=1)": np.diff(y), "seasonal difference (D=1, s=52)": y[52:] - y[:-52],
                "d=1 and D=1": np.diff(y[52:] - y[:-52])}
    fig, axes = plt.subplots(4, 2, figsize=(11, 10))
    for i, (name, v) in enumerate(variants.items()):
        plot_acf(v, lags=60, ax=axes[i, 0], color=PALETTE[0], vlines_kwargs={"colors": PALETTE[0]}, title=f"ACF: {name}")
        plot_pacf(v, lags=60, ax=axes[i, 1], method="ywm", color=PALETTE[0], vlines_kwargs={"colors": PALETTE[0]}, title=f"PACF: {name}")
    fig.tight_layout()
    save(fig, p(f"{FIG}/04_acf_pacf_total.png"))
    from statsmodels.tsa.stattools import acf, pacf
    out["acf_total"] = {k: {str(l): float(acf(v, nlags=60)[l]) for l in [1, 2, 4, 13, 26, 51, 52, 53]} for k, v in variants.items()}
    out["pacf_total"] = {k: {str(l): float(pacf(v, nlags=60, method="ywm")[l]) for l in [1, 2, 3, 52]} for k, v in variants.items()}

    # ---------------------------------------------------------------- 5. stationarity tests on aggregate series
    agg_ids = meta[meta.level.isin(["total", "category", "department", "store"])].unique_id
    st_rows = []
    for uid in agg_ids:
        v = pre[uid].to_numpy()
        for tname, tv in {"level": v, "d=1": np.diff(v), "D=1": v[52:] - v[:-52], "d=1,D=1": np.diff(v[52:] - v[:-52])}.items():
            st_rows.append({"unique_id": uid, "transform": tname, **stationarity_tests(tv)})
    stt = pd.DataFrame(st_rows)
    stt.to_csv(p(f"{TAB}/stationarity_tests.csv"), index=False)
    out["stationarity_summary"] = stt.groupby(["transform", "verdict"]).size().unstack(fill_value=0).to_dict()

    # ---------------------------------------------------------------- 6. variance vs level (Box-Cox question)
    lv = pd.DataFrame({"y": y}, index=pre.index)
    yr = lv.groupby(lv.index.year).y.agg(["mean", "std"])
    _, lam = stats.boxcox(y)
    agg_lvl = []
    for uid in meta[meta.level.isin(["store", "department", "store_dept"])].unique_id:
        s = pre[uid]
        g = s.groupby(s.index.year).agg(["mean", "std"])
        g = g[g.index < 2016]
        agg_lvl.append(np.polyfit(np.log(g["mean"]), np.log(g["std"]), 1)[0] if len(g) > 2 else np.nan)
    out["boxcox_total"] = {"yearly_mean": yr["mean"].round(0).to_dict(), "yearly_std": yr["std"].round(0).to_dict(),
                           "mle_lambda": float(lam),
                           "median_loglog_slope_std_on_mean_store_dept": float(np.nanmedian(agg_lvl))}

    # ---------------------------------------------------------------- 7. characteristics for ALL modelled series
    ch_tab = characteristics_table(df, meta, cutoff_week=hold_start)
    ch_tab.to_csv(p("reports/series_characteristics.csv"), index=False)
    out["sbc_by_level"] = ch_tab.groupby(["level", "sbc_class"]).size().unstack(fill_value=0).to_dict("index")
    out["strength_by_level"] = ch_tab.groupby("level")[["trend_strength", "seasonal_strength", "share_zeros", "cv"]].median().round(3).to_dict("index")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lv_order = ["total", "category", "department", "store", "store_dept", "sku_store"]
    for i, lvl in enumerate(lv_order):
        s = ch_tab[ch_tab.level == lvl]
        axes[0].scatter(s.adi, s.cv2, s=16, color=PALETTE[i], label=lvl, alpha=0.85, edgecolor="white", lw=0.4)
        axes[1].scatter(s.trend_strength, s.seasonal_strength, s=16, color=PALETTE[i], label=lvl, alpha=0.85, edgecolor="white", lw=0.4)
    axes[0].axvline(1.32, color=INK2, lw=0.8, ls="--")
    axes[0].axhline(0.49, color=INK2, lw=0.8, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("ADI (avg. inter-demand interval, log)")
    axes[0].set_ylabel("CV² of non-zero demand")
    n_clip = int((ch_tab.cv2 > 2).sum())
    axes[0].set_ylim(-0.05, 2.0)
    for txt, xy in [("smooth", (0.02, 0.03)), ("erratic", (0.02, 0.93)), ("intermittent", (0.80, 0.03)), ("lumpy", (0.85, 0.93))]:
        axes[0].text(*xy, txt, color=INK2, fontsize=9, transform=axes[0].transAxes, fontweight="bold")
    if n_clip:
        axes[0].text(0.30, 0.93, f"({n_clip} erratic points with CV² > 2 not shown)", color=INK2, fontsize=7.5, transform=axes[0].transAxes)
    axes[0].set_title("Syntetos-Boylan demand classes")
    axes[1].set_xlabel("Trend strength F_T")
    axes[1].set_ylabel("Seasonal strength F_S")
    axes[1].set_title("STL strength of trend vs seasonality")
    axes[1].legend(loc="upper left", fontsize=8)
    save(fig, p(f"{FIG}/05_series_characteristics.png"))

    # ---------------------------------------------------------------- 8. zeros: no-sale vs possible stock-out (censoring)
    sku = df[df.unique_id.str.startswith("SKU_")]
    runs = []
    for uid, g in sku.groupby("unique_id"):
        v = g.y.to_numpy()[: max(0, len(g) - 13)]
        z = (v == 0).astype(int)
        # longest run of zeros
        best = cur = 0
        for x in z:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        runs.append({"unique_id": uid, "longest_zero_run": best, "mean_nonzero": v[v > 0].mean() if (v > 0).any() else 0})
    runs = pd.DataFrame(runs).merge(ch_tab[["unique_id", "sbc_class"]], on="unique_id")
    # a run of >= 4 zero weeks for an item that normally sells >= 5 units/week is implausible as pure chance
    runs["suspected_stockout"] = (runs.longest_zero_run >= 4) & (runs.mean_nonzero >= 5)
    runs.to_csv(p(f"{TAB}/sku_zero_runs.csv"), index=False)
    out["suspected_stockouts"] = {"n_sku_with_suspected_stockout_run": int(runs.suspected_stockout.sum()),
                                  "n_sku": int(len(runs)),
                                  "by_class": runs.groupby("sbc_class").suspected_stockout.sum().astype(int).to_dict()}

    # example SKU series (one per class)
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    for ax, cls in zip(axes, ["smooth", "erratic", "intermittent", "lumpy"]):
        uid = ch_tab[(ch_tab.level == "sku_store") & (ch_tab.sbc_class == cls)].sort_values("mean").iloc[-5].unique_id
        g = df[df.unique_id == uid]
        ax.plot(g.ds, g.y, color=PALETTE[0], lw=1.1)
        ax.set_title(f"{cls}: {uid}", fontsize=9.5, loc="left")
    save(fig, p(f"{FIG}/06_sku_examples.png"))

    with open(p("reports/eda_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ["total_yearly_mean", "event_effects", "snap_effect", "price_effect", "boxcox_total",
                                          "stationarity_summary", "sbc_by_level", "strength_by_level", "suspected_stockouts",
                                          "woy_effect_pct_min_max", "dow_index"]}, indent=1, default=str))
    print(outl)


if __name__ == "__main__":
    main()
