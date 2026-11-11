"""Phase 6: refit on ALL 277 weeks and forecast the next 13 weeks (2016-05-21 .. 2016-08-13) with 80/95% intervals.

Usage:  python -m src.forecast_final
Why refit on everything: the holdout has done its job (an honest estimate of accuracy). Throwing away the most
recent 13 weeks at deployment would ignore the most relevant information (current level, latest trend).

Future known-in-advance inputs:
  * SNAP days: each state uses a fixed day-of-month rule (verified on all 1,969 calendar days), extended exactly.
  * Holidays: Independence Day (2016-07-04) is the only Christmas/Thanksgiving/Easter/SuperBowl-type or national
    holiday in the window besides Memorial Day/other events present in calendar.csv (which runs to 2016-06-19).
  * Prices: unknown after the data ends -> last observed price carried forward (assumption: no planned price change).

Model per series = the model with the lowest mean CV MASE at that series' hierarchy level ("selection by level",
decided on CV folds only), with all candidate forecasts kept in outputs/forecasts.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.load import Origin, load_calendar, load_prices, load_series, week_index
from src.features.exog import calendar_regressors
from src.features.lgbm_features import CAT_FEATURES, Panel, feature_names, origin_features, training_frame
from src.models.lightgbm_global import make_model
from src.models.statistical import fit_predict
from src.plotting import INK2, MODEL_COLORS, plt, save, setup
from src.utils import ROOT, load_config, p, set_seed

STAT_MODELS = ["SeasonalNaive", "ETS", "SARIMA", "DHR-ARIMA", "Prophet"]


def future_calendar(cal: pd.DataFrame, raw_cal: pd.DataFrame, n_future: int) -> pd.DataFrame:
    last = cal.ds.iloc[-1]
    rows = []
    snap_days = {"CA": set(range(1, 11)), "TX": {1, 3, 5, 6, 7, 9, 11, 12, 13, 15}, "WI": {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}}
    for k in range(1, n_future + 1):
        ws = last + pd.Timedelta(weeks=k)
        days = pd.date_range(ws, periods=7, freq="D")
        ev = raw_cal[raw_cal.date.isin(days)]
        n_ev = int(ev.event_name_1.notna().sum() + ev.event_name_2.notna().sum()) if len(ev) == 7 else \
            int(any((d.month, d.day) == (7, 4) for d in days))
        rows.append({"ds": ws, "wm_yr_wk": int(cal.wm_yr_wk.iloc[-1]) + k if (int(cal.wm_yr_wk.iloc[-1]) + k) % 100 <= 52 else int(cal.wm_yr_wk.iloc[-1]) + k + 48,
                     "n_events": n_ev, "n_sporting": 0, "n_cultural": 0,
                     "n_national": int(any((d.month, d.day) == (7, 4) for d in days)), "n_religious": 0,
                     "christmas": 0, "thanksgiving": 0, "easter": 0, "superbowl": 0,
                     **{f"snap_{s}": sum(d.day in v for d in days) for s, v in snap_days.items()}})
    return pd.concat([cal, pd.DataFrame(rows)], ignore_index=True)


def main() -> None:
    setup()
    cfg = load_config()
    set_seed(cfg["seed"])
    h = cfg["problem"]["horizon"]
    nW = cfg["data"]["n_full_weeks"]
    df, meta = load_series()
    cal = load_calendar()
    raw_cal = pd.read_csv(ROOT / cfg["paths"]["raw_dir"] / "calendar.csv", parse_dates=["date"])
    calx = future_calendar(cal, raw_cal, h)
    wk_f = np.arange(nW, nW + h)
    Kmap = pd.read_csv(ROOT / "reports/tables/dhr_fourier_K.csv").set_index("unique_id").K.to_dict()
    state = meta.set_index("unique_id").state_id.to_dict()

    # ---------------- statistical models, refit on all data
    def run(uid, g):
        y = g.y.to_numpy(float)
        wk = week_index(g.ds).to_numpy()
        ctx = {"cal_reg": calendar_regressors(state[uid], calx), "cal_ds": calx.ds, "K": Kmap[uid], "seed": cfg["seed"]}
        res = []
        for m in STAT_MODELS:
            o = fit_predict(m, y, wk, wk_f, ctx)
            res.append(pd.DataFrame({"unique_id": uid, "week": wk_f, "h": np.arange(1, h + 1), "model": m, "mean": o["mean"],
                                     "lo80": o["lo80"], "hi80": o["hi80"], "lo95": o["lo95"], "hi95": o["hi95"], "info": o["info"]}))
        return pd.concat(res)
    stat = pd.concat(Parallel(n_jobs=4)(delayed(run)(u, g) for u, g in df.groupby("unique_id")), ignore_index=True)

    # ---------------- LightGBM retrained on every row with target <= last week
    params = json.load(open(ROOT / "reports/tables/lgbm_best_params.json"))
    prices = load_prices()
    last_p = prices.sort_values("ds").groupby("unique_id").tail(1)
    fut_p = pd.concat([last_p.assign(ds=calx.ds.iloc[nW + k]) for k in range(h)])  # price carried forward (assumption)
    pn = Panel(df, meta, calx, pd.concat([prices, fut_p]), nW + h)
    tr = training_frame(pn, nW - 1, range(1, h + 1))
    feats = feature_names(tr)
    mdl = make_model(params, cfg["seed"])
    cap = np.quantile(tr.y_scaled, 0.999)
    mdl.fit(tr[feats], np.minimum(tr.y_scaled, cap), categorical_feature=CAT_FEATURES)
    te = origin_features(pn, nW - 1, range(1, h + 1))
    lg = pd.DataFrame({"unique_id": [pn.ids[i] for i in te.series], "week": te.target_week, "h": te.h, "model": "LightGBM",
                       "mean": np.maximum(mdl.predict(te[feats]) * te.scale.to_numpy(), 0), "scale": te.scale.to_numpy()})
    # empirical intervals from ALL past origins' scaled errors (tuning, CV and holdout), by group and horizon bucket
    past = pd.read_parquet(ROOT / cfg["paths"]["cache_dir"] / "lgbm_forecasts.parquet")
    a = df.assign(week=week_index(df.ds))[["unique_id", "week", "y"]]
    past = past.merge(a, on=["unique_id", "week"])
    past["r"] = (past.y - past["mean"]) / past.scale
    for x in (past, lg):
        x["grp"] = np.where(x.unique_id.str.startswith("SKU_"), "sku", "agg")
        x["hb"] = pd.cut(x.h, [0, 4, 8, 13], labels=["1-4", "5-8", "9-13"]).astype(str)
    q = past.groupby(["grp", "hb"]).r.quantile([0.10, 0.90, 0.025, 0.975]).unstack().reset_index()
    lg = lg.merge(q, on=["grp", "hb"])
    for c, qq in zip(["lo80", "hi80", "lo95", "hi95"], [0.10, 0.90, 0.025, 0.975]):
        lg[c] = np.maximum(lg["mean"] + lg.scale * lg[qq], 0)
    lg["info"] = json.dumps(params)
    lg = lg[stat.columns]

    ens_comp = json.load(open(ROOT / "reports/evaluation_summary.json"))["ensemble_components"]
    e = pd.concat([stat, lg])
    ens = e[e.model.isin(ens_comp)].groupby(["unique_id", "week", "h"])["mean"].mean().reset_index()
    ens["model"] = "Ensemble"
    ens["info"] = f"mean({', '.join(ens_comp)})"
    allf = pd.concat([stat, lg, ens], ignore_index=True)
    allf["ds"] = calx.ds.iloc[allf.week.to_numpy()].to_numpy()

    # ---------------- recommended model per series = best mean CV MASE at its level (decided on CV only)
    lb = pd.read_csv(ROOT / "reports/tables/leaderboard_cv_by_level.csv")
    best_by_level = lb.loc[lb.groupby("level").mean_MASE.idxmin()].set_index("level").model.to_dict()
    lvl = meta.set_index("unique_id").level
    allf["recommended"] = allf.model == allf.unique_id.map(lvl).map(best_by_level)
    allf = allf.merge(meta[["unique_id", "level", "cat_id", "dept_id", "store_id", "state_id"]], on="unique_id")
    cols = ["unique_id", "level", "cat_id", "dept_id", "store_id", "state_id", "ds", "h", "model", "recommended",
            "mean", "lo80", "hi80", "lo95", "hi95", "info"]
    allf[cols].sort_values(["unique_id", "model", "h"]).to_csv(p("outputs/forecasts.csv"), index=False)
    json.dump({"best_model_by_level_cv": best_by_level, "forecast_window": [str(calx.ds.iloc[nW].date()), str(calx.ds.iloc[-1].date())]},
              open(p("reports/final_forecast_summary.json"), "w"), indent=1)

    # ---------------- plots: total, categories, one SKU (recommended model)
    rec = allf[allf.recommended]
    show = ["TOTAL", "CAT_FOODS", "CAT_HOUSEHOLD", "CAT_HOBBIES"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, uid in zip(axes.ravel(), show):
        g = df[df.unique_id == uid].iloc[-104:]
        f = rec[rec.unique_id == uid]
        mname = f.model.iloc[0]
        ax.plot(g.ds, g.y / 1e3, color=INK2, lw=1.2, label="actual")
        ax.fill_between(f.ds, f.lo95 / 1e3, f.hi95 / 1e3, color=MODEL_COLORS[mname], alpha=0.15, lw=0, label="95% interval")
        ax.fill_between(f.ds, f.lo80 / 1e3, f.hi80 / 1e3, color=MODEL_COLORS[mname], alpha=0.30, lw=0, label="80% interval")
        ax.plot(f.ds, f["mean"] / 1e3, color=MODEL_COLORS[mname], lw=1.8, label=f"forecast ({mname})")
        ax.set_title(f"{uid}: next 13 weeks (thousand units)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, p("reports/figures/14_final_forecast.png"))
    s = rec[rec.unique_id.isin(show)].groupby("unique_id").agg(model=("model", "first"), total_13w=("mean", "sum"),
                                                                peak_week=("mean", "idxmax"))
    s["peak_week"] = allf.loc[s.peak_week, "ds"].dt.date.values
    last13 = df[df.unique_id.isin(show)].groupby("unique_id").y.apply(lambda v: v.iloc[-13:].sum())
    ly13 = df[df.unique_id.isin(show)].groupby("unique_id").y.apply(lambda v: v.iloc[-52 + 0: -39].sum())
    s["last_13w_actual"] = last13
    s["same_13w_last_year_actual"] = ly13
    s["vs_same_period_last_year_pct"] = 100 * (s.total_13w / s.same_13w_last_year_actual - 1)
    s.to_csv(p("reports/tables/final_forecast_headline.csv"))
    print(s)
    print(best_by_level)


if __name__ == "__main__":
    main()
