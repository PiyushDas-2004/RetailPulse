"""Leakage-safe feature construction for the global LightGBM model (direct multi-horizon strategy).

One training row = (series i, forecast origin T, horizon h). Target = y[i, T+h] / scale[i, T].

Information rule (tested in tests/test_no_leakage.py):
  * every feature derived from the target uses ONLY y[i, :T+1]  (weeks <= origin)
  * features about the target week T+h use only KNOWN-IN-ADVANCE information: calendar, holidays, the SNAP
    schedule and the planned shelf price (an assumption: a retailer knows its own price plan for the next quarter)
  * seasonal lags y[T+h-52] and y[T+h-104] are <= T because h <= 13 < 52

Why "direct" and not "recursive": a recursive model feeds its own 1-step forecasts back in as lags, so errors
compound over 13 steps and future exogenous features must also be simulated. The direct approach trains on
(origin, horizon) pairs with horizon as a feature: one model, no error feedback, and lags are always real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.exog import snap_for

CAT_FEATURES = ["level_c", "cat_c", "dept_c", "store_c"]


class Panel:
    """Dense (n_series x n_weeks) arrays; NaN before an SKU's first listed week."""

    def __init__(self, df: pd.DataFrame, meta: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame, n_weeks: int,
                 first: str = "2011-01-29"):
        self.ids = meta.unique_id.tolist()
        self.meta = meta.set_index("unique_id").loc[self.ids].reset_index()
        idx = {u: i for i, u in enumerate(self.ids)}
        n = len(self.ids)
        self.n_weeks = n_weeks
        Y = np.full((n, n_weeks), np.nan)
        wk = ((df.ds - pd.Timestamp(first)).dt.days // 7).to_numpy()
        Y[df.unique_id.map(idx).to_numpy(), wk] = df.y.to_numpy()
        self.Y = Y
        P = np.full((n, n_weeks), np.nan)
        if len(prices):
            pw = ((prices.ds - pd.Timestamp(first)).dt.days // 7).to_numpy()
            ok = prices.unique_id.isin(idx).to_numpy()
            P[prices.unique_id[ok].map(idx).to_numpy(), pw[ok]] = prices.price.to_numpy()[ok]
        self.P = P
        self.start = np.argmax(~np.isnan(Y), axis=1)
        self.snap = np.vstack([snap_for(s, cal) for s in self.meta.state_id])
        self.cal = cal.reset_index(drop=True)
        self.woy = (cal.wm_yr_wk % 100).to_numpy()
        self.month = cal.ds.dt.month.to_numpy()
        codes = {}
        for col, src in [("level_c", "level"), ("cat_c", "cat_id"), ("dept_c", "dept_id"), ("store_c", "store_id")]:
            cats = sorted(self.meta[src].astype(str).unique())
            codes[col] = self.meta[src].astype(str).map({c: j for j, c in enumerate(cats)}).to_numpy()
        self.codes = codes


def _nanmean(a, axis=1):
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=1):
    with np.errstate(all="ignore"):
        return np.nanstd(a, axis=axis)


def origin_features(pn: Panel, T: int, horizons, Ymask: np.ndarray | None = None) -> pd.DataFrame:
    """Features for all series at origin T for each h in `horizons`. Uses only pn.Y[:, :T+1]."""
    Y = pn.Y[:, : T + 1]  # <-- the ONLY slice of the target ever read
    n = Y.shape[0]
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    last52 = Y[:, max(0, T - 51): T + 1]
    scale = _nanmean(last52)
    fallback = _nanmean(Y)
    scale = np.where((scale > 0) & np.isfinite(scale), scale, np.where(fallback > 0, fallback, 1.0))
    age = T - pn.start + 1
    base = {
        "scale_log": np.log10(scale), "age": age,
        **{f"lag{k + 1}": Y[:, T - k] / scale for k in range(4)},
        **{f"rmean{w}": _nanmean(Y[:, max(0, T - w + 1): T + 1]) / scale for w in (4, 8, 13, 26, 52)},
        **{f"rstd{w}": _nanstd(Y[:, max(0, T - w + 1): T + 1]) / scale for w in (4, 13, 52)},
        "zero13": _nanmean((Y[:, max(0, T - 12): T + 1] == 0).astype(float)),
        "zero52": _nanmean((Y[:, max(0, T - 51): T + 1] == 0).astype(float)),
    }
    ly13 = _nanmean(Y[:, max(0, T - 64): max(0, T - 51)]) if T >= 64 else np.full(n, np.nan)
    base["yoy13"] = base["rmean13"] * scale / np.where(ly13 > 0, ly13, np.nan)
    xs = np.arange(26) - 12.5
    seg = Y[:, max(0, T - 25): T + 1]
    if seg.shape[1] == 26:
        base["slope26"] = np.nansum((seg - _nanmean(seg)[:, None]) * xs, axis=1) / (xs ** 2).sum() / scale
    else:
        base["slope26"] = np.full(n, np.nan)
    rows = []
    for h in horizons:
        t = T + h
        f = dict(base)
        f["h"] = np.full(n, h)
        f["slag52"] = Y[:, t - 52] / scale if t - 52 <= T else np.nan
        f["slag104"] = Y[:, t - 104] / scale if 0 <= t - 104 else np.full(n, np.nan)
        f["slag52_m3"] = _nanmean(Y[:, max(0, t - 53): t - 50]) / scale
        f["snap_lag52_diff"] = pn.snap[:, t] - pn.snap[:, t - 52]
        # target-week (known in advance) information; guard for t beyond the observed calendar
        tc = min(t, pn.n_weeks - 1)
        cal = pn.cal.iloc[tc] if t < pn.n_weeks else None
        f["woy"] = np.full(n, pn.woy[tc] if t < pn.n_weeks else (pn.woy[T] + h - 1) % 52 + 1)
        f["month"] = np.full(n, pn.month[tc])
        for c in ["christmas", "thanksgiving", "easter", "superbowl", "n_events"]:
            f[c] = np.full(n, float(cal[c]) if cal is not None else np.nan)
        f["snap"] = pn.snap[:, tc] if t < pn.n_weeks else np.full(n, np.nan)
        pt = pn.P[:, tc] if t < pn.n_weeks else pn.P[:, T]
        f["price_rel52"] = pt / _nanmean(pn.P[:, max(0, T - 51): T + 1])
        f["price_chg"] = pt / pn.P[:, T] - 1
        for c, v in pn.codes.items():
            f[c] = v
        d = pd.DataFrame(f)
        d["series"] = np.arange(n)
        d["origin_T"] = T
        d["target_week"] = t
        d["scale"] = scale
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


FEATURES = None  # filled lazily


def feature_names(df: pd.DataFrame) -> list[str]:
    drop = {"series", "origin_T", "target_week", "scale", "y_scaled", "y"}
    return [c for c in df.columns if c not in drop]


def training_frame(pn: Panel, train_end: int, horizons, min_age: int = 52, first_T: int = 52) -> pd.DataFrame:
    """All (series, T, h) rows whose TARGET week is <= train_end. This is what makes the split leakage-free."""
    frames = []
    for T in range(first_T, train_end):
        hs = [h for h in horizons if T + h <= train_end]
        if not hs:
            continue
        d = origin_features(pn, T, hs)
        y = pn.Y[d.series.to_numpy(), d.target_week.to_numpy()]
        d["y_scaled"] = y / d.scale.to_numpy()
        ok = np.isfinite(y) & (d.age.to_numpy() >= min_age)
        frames.append(d[ok])
    out = pd.concat(frames, ignore_index=True)
    assert out.target_week.max() <= train_end, "leakage: a training target lies after the origin"
    return out
