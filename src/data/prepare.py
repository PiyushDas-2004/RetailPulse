"""Validate raw M5, build the weekly modelling dataset, and write the data profile.

Usage:  python -m src.data.prepare

Outputs
  reports/data_profile.json             machine-readable validation profile
  data/processed/weekly_series.parquet  long table: unique_id, ds, y (units/week)
  data/processed/series_meta.csv        one row per modelled series (level, ids, start week, sampling stratum)
  data/processed/weekly_calendar.csv    exogenous calendar features per week (known in advance)
  data/processed/sku_prices.parquet     weekly sell price for the sampled SKU-store series

Design (WHY in DECISIONS.md):
  * weekly buckets = Walmart fiscal weeks (Sat-Fri); 277 complete weeks, final 2 days dropped
  * "not listed" (no price row) is treated as NO DATA, not as zero demand
  * series set = total, 3 categories, 7 departments, 10 stores, 70 store x department,
    plus a stratified sample of item x store series (equal count per Syntetos-Boylan class)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils import ROOT, load_config, p, set_seed


def sbc_class(adi: float, cv2: float) -> str:
    """Syntetos-Boylan-Croston classification with the standard cut-offs (ADI 1.32, CV^2 0.49)."""
    if not np.isfinite(adi):
        return "no_demand"
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


def adi_cv2(y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    nz = y[y > 0]
    if len(nz) == 0:
        return np.inf, np.nan
    adi = len(y) / len(nz)
    cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if len(nz) > 1 else 0.0
    return float(adi), float(cv2)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    raw = ROOT / cfg["paths"]["raw_dir"]
    nW = cfg["data"]["n_full_weeks"]
    nD = nW * 7

    cal = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    if "id" not in sales.columns:  # the Nixtla mirror drops the redundant Kaggle `id`; rebuild it losslessly
        sales.insert(0, "id", sales.item_id + "_" + sales.store_id + "_evaluation")
    prices = pd.read_csv(raw / "sell_prices.csv")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    dcols = [c for c in sales.columns if c.startswith("d_")]
    X = sales[dcols].to_numpy(dtype=np.int32)

    # ------------------------------------------------------------------ profile (daily, raw)
    cal_d = cal.iloc[: len(dcols)]
    prof: dict = {
        "dataset": "M5 Forecasting - Accuracy (Walmart), sales_train_evaluation.csv",
        "files": {
            "sales_train_evaluation.csv": {"rows": int(len(sales)), "cols": int(sales.shape[1])},
            "calendar.csv": {"rows": int(len(cal)), "cols": int(cal.shape[1])},
            "sell_prices.csv": {"rows": int(len(prices)), "cols": int(prices.shape[1])},
        },
        "dtypes": {
            "sales_id_columns": {c: str(sales[c].dtype) for c in id_cols},
            "sales_day_columns": str(X.dtype),
            "calendar": {c: str(t) for c, t in cal.dtypes.items()},
            "sell_prices": {c: str(t) for c, t in prices.dtypes.items()},
        },
        "date_range_daily": [str(cal_d.date.min().date()), str(cal_d.date.max().date())],
        "n_days": len(dcols),
        "n_item_store_series": int(len(sales)),
        "n_items": int(sales.item_id.nunique()),
        "n_stores": int(sales.store_id.nunique()),
        "n_states": int(sales.state_id.nunique()),
        "n_categories": int(sales.cat_id.nunique()),
        "n_departments": int(sales.dept_id.nunique()),
        "missing_values": {
            "sales_cells": int(sales[dcols].isna().sum().sum()),
            "sales_id_cells": int(sales[id_cols].isna().sum().sum()),
            "sell_prices": {c: int(v) for c, v in prices.isna().sum().items()},
            "calendar_event_name_1_non_null_days": int(cal_d.event_name_1.notna().sum()),
        },
        "duplicates": {
            "sales_duplicate_ids": int(sales.id.duplicated().sum()),
            "sales_duplicate_rows_on_id_cols": int(sales.duplicated(subset=id_cols).sum()),
            "prices_duplicate_keys": int(prices.duplicated(subset=["store_id", "item_id", "wm_yr_wk"]).sum()),
            "calendar_duplicate_dates": int(cal.date.duplicated().sum()),
        },
        "negative_values": {"sales_cells": int((X < 0).sum()), "prices": int((prices.sell_price <= 0).sum())},
        "zero_inflation": {
            "share_zero_cells_daily_item_store": float((X == 0).mean()),
        },
        "calendar_gaps_days": int((cal.date.diff().dt.days.dropna() != 1).sum()),
    }
    tot_daily = X.sum(0)
    prof["daily_total_units"] = {
        "min": int(tot_daily.min()),
        "min_date": str(cal_d.date.iloc[int(tot_daily.argmin())].date()),
        "max": int(tot_daily.max()),
        "max_date": str(cal_d.date.iloc[int(tot_daily.argmax())].date()),
        "days_below_10pct_of_median": [str(d.date()) for d in cal_d.date[tot_daily < 0.1 * np.median(tot_daily)]],
    }
    prof["sell_price"] = {k: float(v) for k, v in prices.sell_price.describe().items()}

    # ------------------------------------------------------------------ weekly aggregation
    wk = cal.wm_yr_wk.iloc[:nD].to_numpy().reshape(nW, 7)
    assert (wk == wk[:, :1]).all(), "each 7-day block must be one Walmart week"
    wk_ids = wk[:, 0]
    week_start = cal.date.iloc[:nD:7].reset_index(drop=True)
    assert (week_start.dt.dayofweek == 5).all(), "weeks must start on Saturday"
    W = X[:, :nD].reshape(len(sales), nW, 7).sum(-1).astype(np.int64)  # (series, weeks)
    prof["weekly"] = {
        "n_full_weeks": nW,
        "first_week_start": str(week_start.iloc[0].date()),
        "last_week_start": str(week_start.iloc[-1].date()),
        "days_dropped_at_end": int(len(dcols) - nD),
    }

    # availability (listed) mask from price table: no price row => item not on sale in that store that week
    wk_index = {w: i for i, w in enumerate(wk_ids)}
    pr = prices[prices.wm_yr_wk.isin(wk_index)].copy()
    pr["w"] = pr.wm_yr_wk.map(wk_index)
    key = sales.item_id + "|" + sales.store_id
    row_of = pd.Series(np.arange(len(sales)), index=key.values)
    pr["r"] = row_of.reindex((pr.item_id + "|" + pr.store_id).values).values
    listed = np.zeros_like(W, dtype=bool)
    listed[pr.r.to_numpy(), pr.w.to_numpy()] = True
    P = np.full(W.shape, np.nan)
    P[pr.r.to_numpy(), pr.w.to_numpy()] = pr.sell_price.to_numpy()
    first_listed = np.where(listed.any(1), listed.argmax(1), nW)
    # internal gaps = unlisted weeks after first listing
    after_first = np.arange(nW)[None, :] >= first_listed[:, None]
    gaps = (after_first & ~listed).sum(1)
    prof["listing"] = {
        "sales_in_unlisted_weeks_units": int(W[~listed].sum()),
        "item_store_listed_from_week0": int((first_listed == 0).sum()),
        "item_store_with_internal_unlisted_weeks": int((gaps > 0).sum()),
        "median_first_listed_week": float(np.median(first_listed)),
        "share_item_store_weeks_unlisted_after_launch": float((after_first & ~listed).sum() / after_first.sum()),
    }
    # zeros while listed = genuine "no sale" (or unobserved stock-out); zeros before listing = no data
    listed_zero = (W == 0) & listed
    prof["zero_inflation"]["share_zero_weeks_item_store_while_listed"] = float(listed_zero.sum() / listed.sum())
    prof["zero_inflation"]["share_zero_weeks_item_store_all"] = float((W == 0).mean())

    # ------------------------------------------------------------------ aggregate series
    rows = []
    meta = []
    ds = week_start.to_numpy()

    def add(uid: str, level: str, y: np.ndarray, start: int = 0, **kw):
        rows.append(pd.DataFrame({"unique_id": uid, "ds": ds[start:], "y": y[start:].astype(float)}))
        meta.append({"unique_id": uid, "level": level, "start_week": int(start), "n_weeks": int(nW - start), **kw})

    add("TOTAL", "total", W.sum(0), cat_id="ALL", dept_id="ALL", store_id="ALL", state_id="ALL")
    for c, g in sales.groupby("cat_id").groups.items():
        add(f"CAT_{c}", "category", W[g].sum(0), cat_id=c, dept_id="ALL", store_id="ALL", state_id="ALL")
    for d, g in sales.groupby("dept_id").groups.items():
        add(f"DEPT_{d}", "department", W[g].sum(0), cat_id=d.rsplit("_", 1)[0], dept_id=d, store_id="ALL", state_id="ALL")
    for s, g in sales.groupby("store_id").groups.items():
        add(f"STORE_{s}", "store", W[g].sum(0), cat_id="ALL", dept_id="ALL", store_id=s, state_id=s.split("_")[0])
    for (s, d), g in sales.groupby(["store_id", "dept_id"]).groups.items():
        add(f"SD_{s}_{d}", "store_dept", W[g].sum(0), cat_id=d.rsplit("_", 1)[0], dept_id=d, store_id=s,
            state_id=s.split("_")[0])

    # ------------------------------------------------------------------ SKU (item x store) stratified sample
    min_hist = cfg["data"]["sku_min_history_weeks"]
    holdout_start = nW - cfg["problem"]["horizon"]
    eligible = (nW - first_listed >= min_hist) & (gaps == 0)
    cand = []
    for i in np.where(eligible)[0]:
        y_tr = W[i, first_listed[i]:holdout_start]  # characterise on pre-holdout data only
        adi, cv2 = adi_cv2(y_tr)
        cand.append((i, adi, cv2, sbc_class(adi, cv2), y_tr.mean()))
    cand = pd.DataFrame(cand, columns=["row", "adi", "cv2", "sbc", "mean"])
    prof["sku_sampling"] = {
        "eligible_item_store_series": int(len(cand)),
        "eligibility_rule": f"listed >= {min_hist} of {nW} weeks and no unlisted gap after launch",
        "sbc_counts_in_eligible": cand.sbc.value_counts().to_dict(),
    }
    rng = np.random.default_rng(cfg["seed"])
    k = cfg["data"]["sku_per_class"]
    picks = []
    for cls in ["smooth", "erratic", "intermittent", "lumpy"]:
        c = cand[cand.sbc == cls].copy()
        c["vol_tercile"] = pd.qcut(c["mean"].rank(method="first"), 3, labels=["low", "mid", "high"])
        for t in ["low", "mid", "high"]:
            pool = c[c.vol_tercile == t]
            n = min(len(pool), k // 3)
            picks.append(pool.iloc[rng.choice(len(pool), n, replace=False)])
    picks = pd.concat(picks)
    for _, r in picks.iterrows():
        i = int(r.row)
        s = sales.iloc[i]
        add(f"SKU_{s.item_id}_{s.store_id}", "sku_store", W[i], start=int(first_listed[i]), cat_id=s.cat_id,
            dept_id=s.dept_id, store_id=s.store_id, state_id=s.state_id, item_id=s.item_id,
            sample_stratum=f"{r.sbc}/{r.vol_tercile}")
    long = pd.concat(rows, ignore_index=True)
    meta = pd.DataFrame(meta)
    assert long.groupby("unique_id").size().equals(meta.set_index("unique_id").n_weeks.sort_index().rename(None)
                                                   .reindex(long.groupby("unique_id").size().index)), "length check"
    assert long.y.min() >= 0 and long.y.notna().all()

    # SKU weekly prices (known-in-advance covariate) for the sampled series
    sku_rows = picks.row.astype(int).to_numpy()
    sp = []
    for i in sku_rows:
        s = sales.iloc[i]
        sp.append(pd.DataFrame({"unique_id": f"SKU_{s.item_id}_{s.store_id}", "ds": ds, "price": P[i]}))
    sp = pd.concat(sp, ignore_index=True).dropna()

    # ------------------------------------------------------------------ weekly calendar (exogenous, known in advance)
    c7 = cal.iloc[:nD].copy()
    c7["w"] = np.repeat(np.arange(nW), 7)
    ev = c7.assign(has_event=c7.event_name_1.notna() | c7.event_name_2.notna(),
                   sporting=(c7.event_type_1 == "Sporting") | (c7.event_type_2 == "Sporting"),
                   cultural=(c7.event_type_1 == "Cultural") | (c7.event_type_2 == "Cultural"),
                   national=(c7.event_type_1 == "National") | (c7.event_type_2 == "National"),
                   religious=(c7.event_type_1 == "Religious") | (c7.event_type_2 == "Religious"),
                   christmas=c7.event_name_1.eq("Christmas"),
                   thanksgiving=c7.event_name_1.eq("Thanksgiving"),
                   easter=c7.event_name_1.eq("Easter") | c7.event_name_2.eq("Easter"),
                   superbowl=c7.event_name_1.eq("SuperBowl"))
    wcal = ev.groupby("w").agg(n_events=("has_event", "sum"), n_sporting=("sporting", "sum"),
                               n_cultural=("cultural", "sum"), n_national=("national", "sum"),
                               n_religious=("religious", "sum"), christmas=("christmas", "max"),
                               thanksgiving=("thanksgiving", "max"), easter=("easter", "max"),
                               superbowl=("superbowl", "max"), snap_CA=("snap_CA", "sum"),
                               snap_TX=("snap_TX", "sum"), snap_WI=("snap_WI", "sum"))
    wcal = wcal.astype(int)
    wcal.insert(0, "ds", ds)
    wcal.insert(1, "wm_yr_wk", wk_ids)

    out = ROOT / cfg["paths"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out / "weekly_series.parquet", index=False)
    meta.to_csv(out / "series_meta.csv", index=False)
    wcal.to_csv(out / "weekly_calendar.csv", index=False)
    sp.to_parquet(out / "sku_prices.parquet", index=False)
    prof["modelled_series"] = {
        "n_series": int(len(meta)),
        "by_level": meta.level.value_counts().to_dict(),
        "length_weeks": {k: float(v) for k, v in meta.n_weeks.describe().items()},
    }
    with open(p("reports/data_profile.json"), "w") as fh:
        json.dump(prof, fh, indent=2, default=str)
    print(json.dumps({k: prof[k] for k in ["weekly", "listing", "sku_sampling", "modelled_series"]}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
