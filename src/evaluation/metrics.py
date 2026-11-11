"""Forecast accuracy metrics. All take 1-D arrays of actuals `y` and forecasts `f` over one test window.

WAPE (primary): sum|y-f| / sum|y|  -- scale-free, defined with zeros, volume-weighted.
MASE: MAE / in-sample MAE of the seasonal-naive forecast (lag m) on the training data.
Bias: sum(f-y) / sum(y)  -- positive = over-forecast (overstock), negative = under-forecast (stock-outs).
sMAPE: mean(2|y-f| / (|y|+|f|)), with 0/0 := 0. Reported with a caveat (asymmetric, unstable near zero).
MAPE is deliberately NOT implemented as a reported metric: undefined when y = 0.
"""
from __future__ import annotations

import numpy as np


def mae(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(np.abs(y - f)))


def rmse(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def wape(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    den = np.abs(y).sum()
    return float(np.abs(y - f).sum() / den) if den > 0 else np.nan


def bias(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    den = np.abs(y).sum()
    return float((f - y).sum() / den) if den > 0 else np.nan


def smape(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    den = np.abs(y) + np.abs(f)
    r = np.where(den == 0, 0.0, 2 * np.abs(y - f) / np.where(den == 0, 1, den))
    return float(np.mean(r))


def mase_scale(y_train, m: int = 52) -> float:
    """In-sample MAE of the seasonal-naive forecast. Falls back to lag-1 naive if the seasonal scale is 0/undefined."""
    y = np.asarray(y_train, float)
    if len(y) > m:
        s = np.mean(np.abs(y[m:] - y[:-m]))
        if s > 0:
            return float(s)
    s = np.mean(np.abs(np.diff(y))) if len(y) > 1 else np.nan
    return float(s) if s > 0 else np.nan


def mase(y, f, scale: float) -> float:
    return mae(y, f) / scale if scale and np.isfinite(scale) else np.nan


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def all_metrics(y, f, scale: float) -> dict:
    return {"MAE": mae(y, f), "RMSE": rmse(y, f), "WAPE": wape(y, f), "MASE": mase(y, f, scale),
            "Bias": bias(y, f), "sMAPE": smape(y, f)}
