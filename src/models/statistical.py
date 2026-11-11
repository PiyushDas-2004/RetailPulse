"""Per-series statistical models behind one interface.

    fit_predict(name, y, wk_train, wk_future, ctx) -> dict(mean, lo80, hi80, lo95, hi95, info)

* y          : training values (only weeks <= origin)
* wk_train   : global week indices of y (used for Fourier/calendar regressors)
* wk_future  : global week indices to forecast (origin+1 .. origin+h)
* ctx        : dict with `cal_reg` (DataFrame of known-in-advance regressors for all weeks) and `K` (Fourier order)

Every model sees exactly the same training window; nothing from wk_future except known-in-advance regressors.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import stats

from src.features.exog import fourier

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

M = 52
LEVELS = (80, 95)


def _pack(mean, lo80=None, hi80=None, lo95=None, hi95=None, info=""):
    h = len(mean)
    nan = np.full(h, np.nan)
    return {"mean": np.asarray(mean, float), "lo80": nan if lo80 is None else np.asarray(lo80, float),
            "hi80": nan if hi80 is None else np.asarray(hi80, float), "lo95": nan if lo95 is None else np.asarray(lo95, float),
            "hi95": nan if hi95 is None else np.asarray(hi95, float), "info": info}


def _sf(model, y, h, X=None, Xf=None):
    from statsforecast.models import Naive  # noqa: F401  (import check)
    if X is not None:
        model.fit(y, X)
        fc = model.predict(h, X=Xf, level=list(LEVELS))
    else:
        model.fit(y)
        fc = model.predict(h, level=list(LEVELS))
    return fc


def naive(y, wk_train, wk_future, ctx):
    from statsforecast.models import Naive
    fc = _sf(Naive(), y, len(wk_future))
    return _pack(fc["mean"], fc["lo-80"], fc["hi-80"], fc["lo-95"], fc["hi-95"])


def seasonal_naive(y, wk_train, wk_future, ctx):
    from statsforecast.models import SeasonalNaive
    fc = _sf(SeasonalNaive(season_length=M), y, len(wk_future))
    return _pack(fc["mean"], fc["lo-80"], fc["hi-80"], fc["lo-95"], fc["hi-95"])


def moving_average(y, wk_train, wk_future, ctx):
    w = ctx.get("ma_window", 4)
    return _pack(np.repeat(np.mean(y[-w:]), len(wk_future)), info=f"window={w}")


def ses(y, wk_train, wk_future, ctx):
    from statsforecast.models import SimpleExponentialSmoothingOptimized
    m = SimpleExponentialSmoothingOptimized()
    m.fit(y)
    return _pack(m.predict(len(wk_future))["mean"])


def sarima(y, wk_train, wk_future, ctx):
    """Seasonal ARIMA, orders chosen by the Hyndman-Khandakar stepwise AICc search (statsforecast AutoARIMA).
    d by KPSS tests, D by the seasonal-strength test (F_S > 0.64); approximation=True as R's auto.arima does by
    default for long / high-period series (CSS during search, exact ML for the final model)."""
    from statsforecast.arima import arima_string
    from statsforecast.models import AutoARIMA
    m = AutoARIMA(season_length=M, approximation=True)
    fc = _sf(m, y, len(wk_future))
    return _pack(fc["mean"], fc["lo-80"], fc["hi-80"], fc["lo-95"], fc["hi-95"], info=arima_string(m.model_).strip())


def _dhr_X(wk, K, cal_reg):
    return np.column_stack([fourier(wk, K), cal_reg.iloc[wk].to_numpy()])


def select_fourier_K(y, wk_train, cal_reg, Ks=(2, 4, 6, 8, 10)) -> tuple[int, dict]:
    from statsforecast.models import AutoARIMA
    scores = {}
    for K in Ks:
        try:
            m = AutoARIMA(season_length=1).fit(y, _dhr_X(wk_train, K, cal_reg))
            scores[K] = float(m.model_["aicc"])
        except Exception:  # noqa: BLE001
            scores[K] = np.inf
    return min(scores, key=scores.get), scores


def dhr_arimax(y, wk_train, wk_future, ctx):
    """Dynamic harmonic regression: y_t = Fourier(K terms, period 52.18) + holiday dummies + SNAP days + ARIMA errors.
    Handles the long, non-integer annual period cheaply and adds known-in-advance drivers (the 'X' in SARIMAX)."""
    from statsforecast.arima import arima_string
    from statsforecast.models import AutoARIMA
    K, cal_reg = ctx["K"], ctx["cal_reg"]
    m = AutoARIMA(season_length=1)
    fc = _sf(m, y, len(wk_future), _dhr_X(wk_train, K, cal_reg), _dhr_X(wk_future, K, cal_reg))
    return _pack(fc["mean"], fc["lo-80"], fc["hi-80"], fc["lo-95"], fc["hi-95"], info=f"K={K} " + arima_string(m.model_).strip())


def ets(y, wk_train, wk_future, ctx):
    """ETS (error, trend, seasonal) state-space model, form chosen by AICc.
    statsforecast's AutoETS silently drops seasonality for m > 24, so statsmodels ETSModel is used.
    Seasonal initial states use the heuristic (classical decomposition of the first 2 years) initialisation;
    to keep AICc comparable between seasonal and non-seasonal candidates, the 51 free initial seasonal states are
    added to the parameter count."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    h = len(wk_future)
    pos = y.min() > 0
    cands = [("add", None, None, False), ("add", "add", None, True), ("add", None, "add", False), ("add", "add", "add", True)]
    if pos:
        cands += [("mul", None, "mul", False), ("mul", "add", "mul", True)]
    best, best_aicc, best_spec = None, np.inf, None
    n = len(y)
    for e, t, s, dmp in cands:
        try:
            mod = ETSModel(y, error=e, trend=t, seasonal=s, damped_trend=dmp, seasonal_periods=M if s else None,
                           initialization_method="heuristic" if s else "estimated")
            r = mod.fit(disp=False, maxiter=200)
            k = len(r.params) + (M - 1 if s else 0)
            aicc = -2 * r.llf + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
            if np.isfinite(aicc) and aicc < best_aicc:
                best, best_aicc, best_spec = r, aicc, (e, t, s, dmp)
        except Exception:  # noqa: BLE001
            continue
    mean = best.forecast(h)
    # analytic-or-simulated intervals from statsmodels
    out = {}
    for lvl in LEVELS:
        try:
            sf = best.get_prediction(start=n, end=n + h - 1).summary_frame(alpha=1 - lvl / 100)
            out[lvl] = (sf["pi_lower"].to_numpy(), sf["pi_upper"].to_numpy())
        except Exception:  # noqa: BLE001
            se = np.std(best.resid) * np.sqrt(np.arange(1, h + 1))
            z = stats.norm.ppf(0.5 + lvl / 200)
            out[lvl] = (mean - z * se, mean + z * se)
    e, t, s, dmp = best_spec
    name = f"ETS({e[0].upper()},{'N' if t is None else ('Ad' if dmp else 'A')},{'N' if s is None else s[0].upper()})"
    return _pack(np.asarray(mean), out[80][0], out[80][1], out[95][0], out[95][1], info=name)


def prophet(y, wk_train, wk_future, ctx):
    """Prophet: piecewise-linear trend + yearly Fourier seasonality + holiday-week effects + SNAP regressor."""
    import pandas as pd
    from prophet import Prophet
    cal_reg, cal_ds = ctx["cal_reg"], ctx["cal_ds"]
    hol = []
    for col in ["christmas", "thanksgiving", "easter", "superbowl"]:
        d = cal_ds[cal_reg[col].to_numpy() > 0]
        hol.append(pd.DataFrame({"holiday": col, "ds": d}))
    hol = pd.concat(hol)
    np.random.seed(ctx.get("seed", 42))
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, holidays=hol,
                uncertainty_samples=400, interval_width=0.8)
    m.add_regressor("snap_days")
    tr = pd.DataFrame({"ds": cal_ds.iloc[wk_train].values, "y": y, "snap_days": cal_reg.snap_days.iloc[wk_train].values})
    m.fit(tr)
    fu = pd.DataFrame({"ds": cal_ds.iloc[wk_future].values, "snap_days": cal_reg.snap_days.iloc[wk_future].values})
    mean = m.predict(fu)["yhat"].to_numpy()
    samp = m.predictive_samples(fu)["yhat"]
    q = np.percentile(samp, [10, 90, 2.5, 97.5], axis=1)
    return _pack(mean, q[0], q[1], q[2], q[3])


MODELS = {
    "Naive": naive,
    "SeasonalNaive": seasonal_naive,
    "MovingAverage": moving_average,
    "SES": ses,
    "ETS": ets,
    "SARIMA": sarima,
    "DHR-ARIMA": dhr_arimax,
    "Prophet": prophet,
}


def fit_predict(name, y, wk_train, wk_future, ctx) -> dict:
    out = MODELS[name](np.asarray(y, float), np.asarray(wk_train), np.asarray(wk_future), ctx)
    # demand cannot be negative: clip point forecasts and interval bounds at 0 (applied identically to all models)
    for k in ["mean", "lo80", "hi80", "lo95", "hi95"]:
        out[k] = np.where(np.isnan(out[k]), np.nan, np.maximum(out[k], 0))
    return out
