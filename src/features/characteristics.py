"""Per-series characteristics used to explain *why* a model wins on a series.

Trend / seasonal strength follow Hyndman & Athanasopoulos (FPP3, sec. 4.3) on a robust STL decomposition:
    F_T = max(0, 1 - Var(R) / Var(T + R)),   F_S = max(0, 1 - Var(R) / Var(S + R))
ADI / CV^2 follow Syntetos, Boylan & Croston (2005).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.data.prepare import adi_cv2, sbc_class


def stl_strengths(y: np.ndarray, period: int = 52) -> tuple[float, float]:
    y = np.asarray(y, float)
    if len(y) < 2 * period + 1 or np.allclose(y, y[0]):
        return np.nan, np.nan
    res = STL(y, period=period, robust=True).fit()
    r, t, s = res.resid, res.trend, res.seasonal
    ft = max(0.0, 1 - np.var(r) / np.var(t + r))
    fs = max(0.0, 1 - np.var(r) / np.var(s + r))
    return float(ft), float(fs)


def characterise(y: np.ndarray, period: int = 52) -> dict:
    y = np.asarray(y, float)
    adi, cv2 = adi_cv2(y)
    ft, fs = stl_strengths(y, period)
    # trend direction: relative change of the mean between the last and the first year
    slope = (y[-period:].mean() - y[:period].mean()) / max(y[:period].mean(), 1e-9) if len(y) >= 2 * period else np.nan
    return {
        "n_weeks": len(y), "mean": y.mean(), "std": y.std(ddof=1), "cv": y.std(ddof=1) / y.mean() if y.mean() > 0 else np.nan,
        "share_zeros": float((y == 0).mean()), "adi": adi, "cv2": cv2, "sbc_class": sbc_class(adi, cv2),
        "trend_strength": ft, "seasonal_strength": fs, "yoy_level_change": slope,
    }


def characteristics_table(df: pd.DataFrame, meta: pd.DataFrame, cutoff_week: int, first: str = "2011-01-29") -> pd.DataFrame:
    """Characteristics on data strictly before `cutoff_week` (global week index) -- no peeking at the holdout."""
    wk = (df.ds - pd.Timestamp(first)).dt.days // 7
    pre = df[wk < cutoff_week]
    rows = []
    for uid, g in pre.groupby("unique_id", sort=False):
        rows.append({"unique_id": uid, **characterise(g.y.to_numpy())})
    out = pd.DataFrame(rows).merge(meta[["unique_id", "level", "cat_id", "dept_id", "store_id"]], on="unique_id")
    out["seasonality_band"] = pd.cut(out.seasonal_strength, [-0.01, 0.3, 0.6, 1.0], labels=["weak(<0.3)", "moderate(0.3-0.6)", "strong(>0.6)"])
    out["trend_band"] = pd.cut(out.trend_strength, [-0.01, 0.5, 0.8, 1.0], labels=["weak(<0.5)", "moderate(0.5-0.8)", "strong(>0.8)"])
    return out
