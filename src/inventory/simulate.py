"""Phase 7: turn forecasts into replenishment decisions and back-test them (SIMULATION, not real savings).

Usage:  python -m src.inventory.simulate

Set-up (all assumptions, stated in config/config.yaml and REPORT.md section 12):
  * Policy: weekly periodic review (R = 1), order-up-to level S_t, lost sales, lead time L in {1,2,3,4} weeks
    (M5 has no lead-time data; L is an ASSUMPTION, hence the sensitivity grid), service level in {90,95,98}%.
  * Back-test window: the 13 holdout weeks (264-276). Every week the forecast is UPDATED with the latest actual
    (rolling origin, horizon 1..L+1) but model parameters are NOT re-estimated (as a weekly replenishment run would do):
      - Naive / SeasonalNaive: recomputed from data up to the review week
      - SARIMA / DHR-ARIMA: model fitted at the holdout origin, then `forward()`-filtered on the new data
      - LightGBM: model trained at the holdout origin, features rebuilt from data up to the review week
      - Ensemble: mean of its components
  * sigma = RMSE of the model's own CV-fold errors at horizons 1-4 (16 errors per series). A policy that forecasts
    better is therefore rewarded twice: better cycle stock and a smaller buffer.
  * Extra policy "<best> + demand-sigma": same forecasts, but sigma = std of weekly demand (last 52 pre-holdout
    weeks) to show why safety stock must be sized on forecast error, not on demand variability.
  * Costs: holding 1 per unit-week, shortage 9 per lost unit (assumed ratio; critical ratio 0.9).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.load import load_calendar, load_prices, load_series, make_origins, week_index
from src.features.exog import calendar_regressors
from src.features.lgbm_features import CAT_FEATURES, Panel, feature_names, origin_features, training_frame
from src.forecast_final import future_calendar
from src.inventory.policy import safety_stock, simulate_lost_sales
from src.models.lightgbm_global import make_model
from src.models.statistical import _dhr_X
from src.plotting import INK2, MODEL_COLORS, PALETTE, plt, save, setup
from src.utils import ROOT, load_config, p, set_seed

MAXP = 5  # max protection interval L + R


def rolling_stat(uid, y_all, wk_all, state, K, calx, hold, models):
    """Rolling forecasts for review weeks t = hold.test_start..hold.test_end, origin T = t-1, horizons 1..MAXP."""
    import warnings
    warnings.filterwarnings("ignore")
    from statsforecast.models import AutoARIMA
    out = []
    tr = wk_all <= hold.train_end
    cal_reg = calendar_regressors(state, calx)
    fitted = {}
    if "SARIMA" in models:
        fitted["SARIMA"] = AutoARIMA(season_length=52, approximation=True).fit(y_all[tr])
    if "DHR-ARIMA" in models:
        fitted["DHR-ARIMA"] = AutoARIMA(season_length=1).fit(y_all[tr], _dhr_X(wk_all[tr], K, cal_reg))
    for T in range(hold.train_end, hold.test_end):
        m = wk_all <= T
        y, wk = y_all[m], wk_all[m]
        wk_f = np.arange(T + 1, T + 1 + MAXP)
        f = {"Naive": np.repeat(y[-1], MAXP),
             "SeasonalNaive": np.array([y_all[wk_all == w - 52][0] for w in wk_f])}
        if "SARIMA" in models:
            f["SARIMA"] = fitted["SARIMA"].forward(y=y, h=MAXP)["mean"]
        if "DHR-ARIMA" in models:
            f["DHR-ARIMA"] = fitted["DHR-ARIMA"].forward(y=y, h=MAXP, X=_dhr_X(wk, K, cal_reg), X_future=_dhr_X(wk_f, K, cal_reg))["mean"]
        for name, v in f.items():
            out.append(pd.DataFrame({"unique_id": uid, "origin_T": T, "week": wk_f, "k": np.arange(1, MAXP + 1), "model": name,
                                     "mean": np.maximum(np.asarray(v, float), 0)}))
    return pd.concat(out)


def main() -> None:
    setup()
    cfg = load_config()
    set_seed(cfg["seed"])
    inv = cfg["inventory"]
    nW = cfg["data"]["n_full_weeks"]
    hold = [o for o in make_origins(cfg) if o.name == "holdout"][0]
    df, meta = load_series()
    cal = load_calendar()
    raw_cal = pd.read_csv(ROOT / cfg["paths"]["raw_dir"] / "calendar.csv", parse_dates=["date"])
    calx = future_calendar(cal, raw_cal, MAXP + 1)
    ev = json.load(open(ROOT / "reports/evaluation_summary.json"))
    ens_comp = ev["ensemble_components"]
    lb = pd.read_csv(ROOT / "reports/tables/leaderboard_cv_by_group.csv")
    # "best" model for replenishment = lowest mean CV MASE on the group being replenished (decided on CV only)
    best = {g: lb[lb.group == g].sort_values("mean_MASE").model.iloc[0] for g in lb.group.unique()}
    need = sorted({"SARIMA", "DHR-ARIMA"} & (set(ens_comp) | set(best.values()) | {"SARIMA"}))
    ids = meta[meta.level.isin(["sku_store", "store_dept"])].unique_id.tolist()
    Kmap = pd.read_csv(ROOT / "reports/tables/dhr_fourier_K.csv").set_index("unique_id").K.to_dict()
    state = meta.set_index("unique_id").state_id.to_dict()
    groups = {u: (g.y.to_numpy(float), week_index(g.ds).to_numpy()) for u, g in df.groupby("unique_id")}

    roll = pd.concat(Parallel(n_jobs=4)(delayed(rolling_stat)(u, *groups[u], state[u], Kmap[u], calx, hold, need) for u in ids),
                     ignore_index=True)

    # LightGBM: trained once at the holdout origin, features refreshed weekly
    params = json.load(open(ROOT / "reports/tables/lgbm_best_params.json"))
    prices = load_prices()
    last_p = prices.sort_values("ds").groupby("unique_id").tail(1)
    fut_p = pd.concat([last_p.assign(ds=calx.ds.iloc[nW + k]) for k in range(MAXP + 1)])
    pn = Panel(df, meta, calx, pd.concat([prices, fut_p]), nW + MAXP + 1)
    pn.Y[:, nW:] = np.nan
    tr = training_frame(pn, hold.train_end, range(1, cfg["problem"]["horizon"] + 1))
    feats = feature_names(tr)
    mdl = make_model(params, cfg["seed"])
    mdl.fit(tr[feats], np.minimum(tr.y_scaled, np.quantile(tr.y_scaled, 0.999)), categorical_feature=CAT_FEATURES)
    lg = []
    for T in range(hold.train_end, hold.test_end):
        te = origin_features(pn, T, range(1, MAXP + 1))
        lg.append(pd.DataFrame({"unique_id": [pn.ids[i] for i in te.series], "origin_T": T, "week": te.target_week,
                                "k": te.h, "model": "LightGBM", "mean": np.maximum(mdl.predict(te[feats]) * te.scale.to_numpy(), 0)}))
    roll = pd.concat([roll, pd.concat(lg)], ignore_index=True)
    roll = roll[roll.unique_id.isin(ids)]
    ens = roll[roll.model.isin(ens_comp)].groupby(["unique_id", "origin_T", "week", "k"])["mean"].mean().reset_index().assign(model="Ensemble")
    roll = pd.concat([roll, ens], ignore_index=True)
    roll.to_parquet(ROOT / cfg["paths"]["cache_dir"] / "inventory_rolling_forecasts.parquet", index=False)

    # sigma per series and model from CV errors at horizons 1-4
    allf = pd.read_parquet(ROOT / cfg["paths"]["cache_dir"] / "all_forecasts.parquet")
    cvf = allf[allf.origin.str.startswith("cv") & (allf.h <= 4)]
    sig = cvf.assign(e2=(cvf["mean"] - cvf.y) ** 2).groupby(["unique_id", "model"]).e2.mean().pow(0.5).rename("sigma").reset_index()
    dem_sig = {u: np.std(groups[u][0][(groups[u][1] <= hold.train_end)][-52:], ddof=1) for u in ids}

    policies = {"Naive": "Naive", "SeasonalNaive": "SeasonalNaive"}
    for g, mname in best.items():
        policies[mname] = mname
    policies.update({m: m for m in ["SARIMA", "LightGBM", "Ensemble"]})
    test_weeks = np.arange(hold.test_start, hold.test_end + 1)
    R = inv["review_period_weeks"]
    rows = []
    rl = roll.set_index(["unique_id", "model", "origin_T", "k"])["mean"].sort_index()
    sigd = sig.set_index(["unique_id", "model"]).sigma
    for uid in ids:
        y_all, wk_all = groups[uid]
        dem = np.array([y_all[wk_all == w][0] for w in test_weeks])
        for pol, mname in list(policies.items()) + [(f"{best['SKU (120)'] if uid.startswith('SKU') else best['Aggregates (91)']} + demand-sigma", "DEMAND_SIGMA")]:
            fm = mname if mname != "DEMAND_SIGMA" else (best["SKU (120)"] if uid.startswith("SKU") else best["Aggregates (91)"])
            sigma = dem_sig[uid] if mname == "DEMAND_SIGMA" else sigd.get((uid, fm), np.nan)
            F = np.vstack([rl.loc[(uid, fm, T)].to_numpy() for T in range(hold.train_end, hold.test_end)])  # (13, MAXP)
            for L in inv["lead_times_weeks"]:
                for sl in inv["service_levels"]:
                    ss = safety_stock(sigma, L, sl, R)
                    S = F[:, : L + R].sum(1) + ss
                    init_pipe = F[0, :L]  # orders placed before the window, sized to the forecast at the first origin
                    r = simulate_lost_sales(dem, S, L, initial_on_hand=ss, initial_pipeline=init_pipe,
                                            holding_cost=inv["holding_cost_per_unit_week"], shortage_cost=inv["shortage_cost_per_unit"])
                    rows.append({"unique_id": uid, "group": "SKU" if uid.startswith("SKU") else "store_dept", "policy": pol,
                                 "forecast_model": fm, "lead_time": L, "service_level": sl, "sigma": sigma, "safety_stock": ss,
                                 "mean_weekly_demand": dem.mean(), **r})
    res = pd.DataFrame(rows)
    res.to_csv(p("reports/tables/inventory_simulation_by_series.csv"), index=False)

    agg = res.groupby(["group", "policy", "lead_time", "service_level"]).agg(
        n_series=("unique_id", "size"), stockout_rate=("stockout_rate", "mean"), lost_units=("lost_units", "sum"),
        demand=("demand", "sum"), avg_on_hand=("avg_on_hand", "sum"), holding_cost=("holding_cost", "sum"),
        shortage_cost=("shortage_cost", "sum"), total_cost=("total_cost", "sum"), safety_stock=("safety_stock", "sum")).reset_index()
    agg["fill_rate"] = 1 - agg.lost_units / agg.demand
    agg["weeks_of_cover_on_hand"] = agg.avg_on_hand / (agg.demand / len(test_weeks))
    base = agg[agg.policy == "SeasonalNaive"].set_index(["group", "lead_time", "service_level"])
    for c in ["total_cost", "avg_on_hand", "lost_units", "stockout_rate"]:
        agg[f"{c}_vs_SeasonalNaive_pct"] = 100 * (agg.set_index(["group", "lead_time", "service_level"])[c].values /
                                                  base.reindex(agg.set_index(["group", "lead_time", "service_level"]).index)[c].values - 1)
    agg.to_csv(p("reports/tables/inventory_simulation_summary.csv"), index=False)
    dflt = agg[(agg.lead_time == inv["default_lead_time_weeks"]) & (agg.service_level == inv["default_service_level"])]
    print(dflt.round(3).to_string(index=False))

    # ---------------- recommendations for the NEXT replenishment cycle (from final forecasts) + at-risk list
    ff = pd.read_csv(ROOT / "outputs/forecasts.csv", parse_dates=["ds"])
    L, sl = inv["default_lead_time_weeks"], inv["default_service_level"]
    recs = []
    for uid in ids:
        fm = best["SKU (120)"] if uid.startswith("SKU") else best["Aggregates (91)"]
        f = ff[(ff.unique_id == uid) & (ff.model == fm)].sort_values("h")["mean"].to_numpy()
        sigma = sigd.get((uid, fm), np.nan)
        ss = safety_stock(sigma, L, sl, 0)
        recs.append({"unique_id": uid, "forecast_model": fm, "lead_time_weeks": L, "service_level": sl,
                     "forecast_next_week": f[0], "forecast_lead_time_demand": f[:L].sum(), "sigma_forecast_error": sigma,
                     "safety_stock": ss, "reorder_point": f[:L].sum() + ss,
                     "order_up_to_level": f[: L + R].sum() + safety_stock(sigma, L, sl, R),
                     "safety_stock_weeks_of_cover": ss / max(f[: L].mean(), 1e-9)})
    recs = pd.DataFrame(recs)
    simd = res[(res.policy == fm) | (res.forecast_model == res.policy)]
    simd = res[(res.lead_time == L) & (res.service_level == sl) & (res.policy == res.forecast_model)]
    simd = simd[simd.policy == simd.unique_id.map(lambda u: best["SKU (120)"] if u.startswith("SKU") else best["Aggregates (91)"])]
    cvb = pd.read_csv(ROOT / "reports/tables/series_metrics_cv.csv")
    cvb = cvb.set_index(["unique_id", "model"]).Bias
    recs["cv_bias"] = [cvb.get((u, m), np.nan) for u, m in zip(recs.unique_id, recs.forecast_model)]
    recs = recs.merge(simd[["unique_id", "stockout_rate", "fill_rate", "avg_on_hand", "mean_weekly_demand"]].rename(
        columns={"stockout_rate": "sim_stockout_rate", "fill_rate": "sim_fill_rate", "avg_on_hand": "sim_avg_on_hand"}), on="unique_id")
    recs["sim_weeks_of_cover"] = recs.sim_avg_on_hand / recs.mean_weekly_demand.replace(0, np.nan)
    ch = pd.read_csv(ROOT / "reports/series_characteristics.csv")[["unique_id", "sbc_class", "cv"]]
    recs = recs.merge(ch, on="unique_id")
    recs["risk"] = np.select(
        [(recs.sim_fill_rate < 0.90) | (recs.cv_bias < -0.10), (recs.sim_weeks_of_cover > 3) | (recs.cv_bias > 0.10)],
        ["STOCK-OUT RISK", "OVERSTOCK RISK"], "ok")
    recs["risk_score"] = np.where(recs.risk == "STOCK-OUT RISK", (1 - recs.sim_fill_rate) + recs.cv_bias.clip(upper=0).abs(),
                                  np.where(recs.risk == "OVERSTOCK RISK", recs.sim_weeks_of_cover / 10 + recs.cv_bias.clip(lower=0), 0))
    recs.sort_values(["risk", "risk_score"], ascending=[True, False]).to_csv(p("outputs/inventory_recommendations.csv"), index=False)
    print(recs.risk.value_counts())

    # ---------------- figures: sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sub = agg[(agg.group == "SKU") & (agg.service_level == sl)]
    pols = [x for x in ["Naive", "SeasonalNaive", "SARIMA", "LightGBM", "Ensemble"] if x in sub.policy.unique()]
    for ax, c, lab in zip(axes, ["total_cost", "fill_rate", "avg_on_hand"], ["Total cost proxy (sum over 120 SKUs)",
                                                                            "Fill rate", "Avg units on hand (sum over SKUs)"]):
        for pol in pols:
            d = sub[sub.policy == pol].sort_values("lead_time")
            ax.plot(d.lead_time, d[c], marker="o", ms=5, color=MODEL_COLORS.get(pol, INK2), label=pol)
        ax.set_xlabel("Lead time (weeks, assumed)")
        ax.set_title(lab, fontsize=10)
        ax.set_xticks(inv["lead_times_weeks"])
    axes[0].legend(fontsize=8)
    fig.suptitle(f"SIMULATED replenishment back-test on the 13-week holdout, SKU level, service level {int(sl*100)}%", fontweight="bold")
    fig.tight_layout()
    save(fig, p("reports/figures/15_inventory_sensitivity.png"))
    json.dump({"best_model_by_group": best, "policies": list(policies) + ["<best> + demand-sigma"]},
              open(p("reports/inventory_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
