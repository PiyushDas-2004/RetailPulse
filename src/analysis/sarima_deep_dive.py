"""SARIMA deep dive on the TOTAL series: differencing, transform, AICc grid, CV check, diagnostics, coverage.

Usage:  python -m src.analysis.sarima_deep_dive
Order IDENTIFICATION uses only weeks 0-211 (the training data of CV fold 1), so the CV folds and the holdout are
genuinely out-of-sample for the order choice. Parameters are re-estimated at every origin with the orders fixed.
Engine: statsforecast's ARIMA (same code as AutoARIMA). statsmodels' SARIMAX with s=52 carries a ~110-dimensional
state vector and was impractically slow here (see REPORT section 8).
"""
from __future__ import annotations

import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsforecast.arima import arima_string, ndiffs, nsdiffs
from statsforecast.models import ARIMA, AutoARIMA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.data.load import load_series, make_origins
from src.evaluation.metrics import mase_scale
from src.features.characteristics import stl_strengths
from src.plotting import INK2, PALETTE, plt, save, setup
from src.utils import load_config, p

warnings.filterwarnings("ignore")
UID = "TOTAL"


def fit(y, order, sorder, log=False):
    m = ARIMA(order=order, season_length=52, seasonal_order=sorder, include_mean=False, method="CSS-ML")
    m.fit(np.log(y) if log else y)
    return m


def forecast(m, h, log=False, sigma2_adj=True):
    fc = m.predict(h, level=[80, 95])
    if not log:
        return fc
    out = {k: np.exp(v) for k, v in fc.items()}
    if sigma2_adj:  # bias-adjusted back-transform of the mean (median -> mean)
        out["mean"] = out["mean"] * np.exp(0.5 * m.model_["sigma2"])
    return out


def evaluate_orders(y, origins, order, sorder, log=False):
    rows = []
    for o in origins:
        tr = y[: o.train_end + 1]
        te = y[o.test_start: o.test_end + 1]
        fc = forecast(fit(tr, order, sorder, log), len(te), log)
        f = fc["mean"]
        rows.append({"origin": o.name, "MAE": np.mean(np.abs(te - f)), "WAPE": np.abs(te - f).sum() / te.sum(),
                     "MASE": np.mean(np.abs(te - f)) / mase_scale(tr, 52), "Bias": (f - te).sum() / te.sum(),
                     "cov80": np.mean((te >= fc["lo-80"]) & (te <= fc["hi-80"])),
                     "cov95": np.mean((te >= fc["lo-95"]) & (te <= fc["hi-95"]))})
    return pd.DataFrame(rows)


def main() -> None:
    setup()
    cfg = load_config()
    df, _ = load_series()
    y = df[df.unique_id == UID].y.to_numpy(float)
    origins = [o for o in make_origins(cfg) if o.kind != "tuning"]
    ident = y[: origins[0].train_end + 1]  # weeks 0..211
    out: dict = {"identification_window_weeks": len(ident)}

    # 1. differencing orders from tests (same tests AutoARIMA uses)
    out["ndiffs_kpss"] = int(ndiffs(ident, test="kpss"))
    out["nsdiffs_seas"] = int(nsdiffs(ident, period=52, test="seas"))
    out["stl_strengths_ident"] = dict(zip(["trend", "seasonal"], stl_strengths(ident)))
    d, D = max(out["ndiffs_kpss"], 1), out["nsdiffs_seas"]
    out["chosen_d_D"] = [d, D]

    # 2. AICc grid over a small, justified space (max 3 AR, 2 MA, 1 seasonal AR/MA)
    grid = []
    t0 = time.time()
    for p_, q, P, Q in itertools.product(range(4), range(3), range(2), range(2)):
        try:
            m = fit(ident, (p_, d, q), (P, D, Q))
            grid.append({"p": p_, "q": q, "P": P, "Q": Q, "aicc": m.model_["aicc"], "aic": m.model_["aic"], "bic": m.model_["bic"],
                         "sigma2": m.model_["sigma2"]})
        except Exception as exc:  # noqa: BLE001
            grid.append({"p": p_, "q": q, "P": P, "Q": Q, "aicc": np.nan, "error": str(exc)[:60]})
    grid = pd.DataFrame(grid).sort_values("aicc").reset_index(drop=True)
    out["grid_seconds"] = round(time.time() - t0, 1)
    out["grid_size"] = len(grid)

    # 3. confirm out-of-sample: top-6 by AICc on CV folds (fixed orders, parameters re-estimated per fold)
    cv_names = [o.name for o in origins if o.kind == "cv"]
    top = grid.dropna(subset=["aicc"]).head(6).copy()
    res = []
    for _, r in top.iterrows():
        ev = evaluate_orders(y, origins, (int(r.p), d, int(r.q)), (int(r.P), D, int(r.Q)))
        res.append({"order": f"({int(r.p)},{d},{int(r.q)})({int(r.P)},{D},{int(r.Q)})[52]", "aicc": r.aicc,
                    "cv_mean_MASE": ev[ev.origin.isin(cv_names)].MASE.mean(), "cv_mean_WAPE": ev[ev.origin.isin(cv_names)].WAPE.mean(),
                    "holdout_WAPE": ev[ev.origin == "holdout"].WAPE.iloc[0]})
    top_cv = pd.DataFrame(res)
    top_cv["aicc_rank"] = range(1, len(top_cv) + 1)
    top_cv["cv_rank"] = top_cv.cv_mean_MASE.rank().astype(int)
    grid.to_csv(p("reports/tables/sarima_total_aicc_grid.csv"), index=False)
    top_cv.to_csv(p("reports/tables/sarima_total_top_aicc_vs_cv.csv"), index=False)
    best = grid.iloc[0]
    order, sorder = (int(best.p), d, int(best.q)), (int(best.P), D, int(best.Q))
    out["aicc_best"] = f"{order}{sorder}[52]"

    # 4. transformation: none vs log (Box-Cox lambda = 0), same orders, judged on CV
    ev_none = evaluate_orders(y, origins, order, sorder, log=False)
    ev_log = evaluate_orders(y, origins, order, sorder, log=True)
    tr_cmp = pd.concat([ev_none.assign(transform="none"), ev_log.assign(transform="log")])
    tr_cmp.to_csv(p("reports/tables/sarima_total_transform_cv.csv"), index=False)
    out["transform_cv_mean_MASE"] = {"none": float(ev_none[ev_none.origin.isin(cv_names)].MASE.mean()),
                                     "log": float(ev_log[ev_log.origin.isin(cv_names)].MASE.mean())}
    out["fold_metrics_chosen"] = ev_none.round(4).to_dict("records")
    out["coverage_chosen_all_windows"] = {"cov80": float(ev_none.cov80.mean()), "cov95": float(ev_none.cov95.mean()),
                                          "n_points": int(13 * len(ev_none))}

    # 5. automatic selection for comparison (exact vs approximation) on the same identification window
    for approx in (True, False):
        t1 = time.time()
        am = AutoARIMA(season_length=52, approximation=approx).fit(ident)
        out[f"autoarima_approx_{approx}"] = {"model": arima_string(am.model_).strip(), "aicc": float(am.model_["aicc"]),
                                             "seconds": round(time.time() - t1, 1)}

    # 6. residual diagnostics: chosen orders fitted on all pre-holdout data (weeks 0..263)
    pre = y[: origins[-1].train_end + 1]
    m = fit(pre, order, sorder)
    resid = np.asarray(m.model_["residuals"])[52 * D + d:]  # drop the burn-in lost to differencing
    k = sum(order[::2]) + sorder[0] + sorder[2]
    lb = acorr_ljungbox(resid, lags=[4, 8, 13, 26, 52], model_df=k, return_df=True)
    lb.index.name = "lag"
    lb.to_csv(p("reports/tables/sarima_total_ljung_box.csv"))
    jb = stats.jarque_bera(resid)
    sh = stats.shapiro(resid)
    out["diagnostics"] = {
        "model": arima_string(m.model_).strip(), "coef": {k_: float(v) for k_, v in m.model_["coef"].items()},
        "n_resid": len(resid), "resid_mean": float(resid.mean()), "resid_sd": float(resid.std()),
        "ljung_box_p": {int(i): float(v) for i, v in lb.lb_pvalue.items()},
        "jarque_bera_p": float(jb.pvalue), "shapiro_p": float(sh.pvalue), "skew": float(stats.skew(resid)),
        "excess_kurtosis": float(stats.kurtosis(resid)),
        "largest_abs_resid_weeks": [str(df[df.unique_id == UID].ds.iloc[52 * D + d + i].date()) for i in np.argsort(-np.abs(resid))[:5]],
    }
    ds = df[df.unique_id == UID].ds.iloc[52 * D + d: origins[-1].train_end + 1].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(ds, resid / 1e3, color=PALETTE[0], lw=1)
    axes[0, 0].axhline(0, color=INK2, lw=0.8)
    axes[0, 0].set_title(f"Residuals of {out['diagnostics']['model']} (thousand units)")
    plot_acf(resid, lags=52, ax=axes[0, 1], color=PALETTE[0], vlines_kwargs={"colors": PALETTE[0]},
             title=f"Residual ACF (Ljung-Box p at lag 13 = {lb.lb_pvalue.loc[13]:.3f}, lag 52 = {lb.lb_pvalue.loc[52]:.3f})")
    axes[1, 0].hist(resid / 1e3, bins=30, color=PALETTE[0], edgecolor="#fcfcfb")
    axes[1, 0].set_title(f"Histogram (skew {stats.skew(resid):.2f}, excess kurtosis {stats.kurtosis(resid):.2f})")
    qqplot(resid, line="s", ax=axes[1, 1], markerfacecolor=PALETTE[0], markeredgecolor=PALETTE[0], markersize=3)
    axes[1, 1].set_title(f"Normal QQ plot (Jarque-Bera p = {jb.pvalue:.3g})")
    fig.tight_layout()
    save(fig, p("reports/figures/09_sarima_residual_diagnostics.png"))

    # 7. forecast vs actual with intervals on each evaluation window
    fig, axes = plt.subplots(1, len(origins), figsize=(16, 3.4), sharey=True)
    for ax, o in zip(axes, origins):
        tr = y[: o.train_end + 1]
        fc = forecast(fit(tr, order, sorder), 13)
        dd = df[df.unique_id == UID].ds.iloc[o.test_start: o.test_end + 1]
        hist = df[df.unique_id == UID].iloc[o.train_end - 25: o.train_end + 1]
        ax.plot(hist.ds, hist.y / 1e3, color=INK2, lw=1)
        ax.plot(dd, y[o.test_start: o.test_end + 1] / 1e3, color=INK2, lw=1.2)
        ax.fill_between(dd, fc["lo-95"] / 1e3, fc["hi-95"] / 1e3, color=PALETTE[0], alpha=0.15, lw=0)
        ax.fill_between(dd, fc["lo-80"] / 1e3, fc["hi-80"] / 1e3, color=PALETTE[0], alpha=0.3, lw=0)
        ax.plot(dd, fc["mean"] / 1e3, color=PALETTE[0], lw=1.6)
        ax.set_title(o.name, fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].set_ylabel("thousand units")
    fig.suptitle(f"TOTAL: SARIMA{out['aicc_best']} forecasts (blue) vs actuals (grey), 80/95% intervals", fontweight="bold")
    fig.tight_layout()
    save(fig, p("reports/figures/09b_sarima_total_windows.png"))

    json.dump(out, open(p("reports/sarima_deep_dive.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    print(top_cv.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
