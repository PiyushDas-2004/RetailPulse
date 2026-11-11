"""Global LightGBM model (one model for all 211 series), direct multi-horizon.

Usage:  python -m src.models.lightgbm_global
Steps
  1. Hyper-parameter tuning on the two TUNING origins only (test windows weeks 186-198 and 199-211), selecting the
     grid point with the lowest mean per-series MASE. CV folds and holdout are never used for tuning.
  2. With the chosen parameters, retrain at every origin (tune1..holdout) on rows whose target week <= origin and
     forecast the next 13 weeks.
  3. Prediction intervals: empirical quantiles of scaled errors from *earlier* origins only (split-conformal style),
     by series group (aggregate vs SKU) and horizon bucket.
Output: outputs/cache/lgbm_forecasts.parquet, reports/tables/lgbm_tuning.csv, reports/tables/lgbm_feature_importance.csv
"""
from __future__ import annotations

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.load import load_calendar, load_prices, load_series, make_origins
from src.evaluation.metrics import mase_scale
from src.features.lgbm_features import CAT_FEATURES, Panel, feature_names, origin_features, training_frame
from src.utils import ROOT, load_config, p, set_seed


def make_model(params: dict, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="regression", learning_rate=params["learning_rate"], num_leaves=params["num_leaves"],
                             min_child_samples=params["min_data_in_leaf"], colsample_bytree=params["feature_fraction"],
                             subsample=0.8, subsample_freq=1, n_estimators=params["n_estimators"], random_state=seed,
                             deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)


def fit_predict_origin(pn: Panel, origin, params: dict, seed: int, h: int, return_model: bool = False):
    tr = training_frame(pn, origin.train_end, range(1, h + 1))
    feats = feature_names(tr)
    m = make_model(params, seed)
    # winsorise training LABELS at their 99.9th percentile: a handful of SKU rows have a near-zero trailing scale
    # (long zero runs, likely stock-outs) giving scaled targets > 20x that would dominate the L2 loss
    cap = np.quantile(tr.y_scaled, 0.999)
    m.fit(tr[feats], np.minimum(tr.y_scaled, cap), categorical_feature=CAT_FEATURES)
    te = origin_features(pn, origin.train_end, range(1, h + 1))
    pred = np.maximum(m.predict(te[feats]) * te.scale.to_numpy(), 0)
    out = pd.DataFrame({"unique_id": [pn.ids[i] for i in te.series], "origin": origin.name, "week": te.target_week,
                        "h": te.h, "model": "LightGBM", "mean": pred, "scale": te.scale.to_numpy()})
    return (out, m, feats, tr) if return_model else out


def series_mase(pn: Panel, fc: pd.DataFrame, train_end: int) -> pd.Series:
    idx = {u: i for i, u in enumerate(pn.ids)}
    res = {}
    for uid, g in fc.groupby("unique_id"):
        i = idx[uid]
        ytr = pn.Y[i, : train_end + 1]
        ytr = ytr[~np.isnan(ytr)]
        y = pn.Y[i, g.week.to_numpy()]
        res[uid] = np.mean(np.abs(y - g["mean"].to_numpy())) / mase_scale(ytr, 52)
    return pd.Series(res)


def add_intervals(fc: pd.DataFrame, pn: Panel, order: list[str]) -> pd.DataFrame:
    """Empirical intervals from scaled errors of strictly earlier origins (none for the first two origins)."""
    idx = {u: i for i, u in enumerate(pn.ids)}
    fc = fc.copy()
    fc["y"] = [pn.Y[idx[u], w] if w < pn.n_weeks else np.nan for u, w in zip(fc.unique_id, fc.week)]
    fc["r"] = (fc.y - fc["mean"]) / fc.scale
    fc["grp"] = np.where(fc.unique_id.str.startswith("SKU_"), "sku", "agg")
    fc["hb"] = pd.cut(fc.h, [0, 4, 8, 13], labels=["1-4", "5-8", "9-13"]).astype(str)
    for c in ["lo80", "hi80", "lo95", "hi95"]:
        fc[c] = np.nan
    for j, o in enumerate(order):
        past = fc[fc.origin.isin(order[:j]) & fc.r.notna()]
        if past.origin.nunique() < 2:
            continue
        q = past.groupby(["grp", "hb"]).r.quantile([0.10, 0.90, 0.025, 0.975]).unstack()
        cur = fc.origin == o
        for (g, hb), qs in q.iterrows():
            m = cur & (fc.grp == g) & (fc.hb == hb)
            for c, qq in zip(["lo80", "hi80", "lo95", "hi95"], [0.10, 0.90, 0.025, 0.975]):
                fc.loc[m, c] = np.maximum(fc.loc[m, "mean"] + fc.loc[m, "scale"] * qs[qq], 0)
    return fc.drop(columns=["y", "r", "grp", "hb"])


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    h = cfg["problem"]["horizon"]
    df, meta = load_series()
    pn = Panel(df, meta, load_calendar(), load_prices(), cfg["data"]["n_full_weeks"])
    origins = make_origins(cfg)
    tuning = [o for o in origins if o.kind == "tuning"]

    # ---- 1. tuning on tuning origins only
    grid = cfg["lightgbm"]["param_grid"]
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    rows = []
    t0 = time.time()
    for params in combos:
        scores = []
        for o in tuning:
            fc = fit_predict_origin(pn, o, params, seed, h)
            scores.append(series_mase(pn, fc, o.train_end).mean())
        rows.append({**params, "mean_MASE_tuning": float(np.mean(scores)), **{f"MASE_{o.name}": s for o, s in zip(tuning, scores)}})
        print(f"[lgbm] {params} -> {np.mean(scores):.4f} ({time.time()-t0:.0f}s)")
    tun = pd.DataFrame(rows).sort_values("mean_MASE_tuning")
    tun.to_csv(p("reports/tables/lgbm_tuning.csv"), index=False)
    best = {k: (int(v) if k in ("num_leaves", "min_data_in_leaf", "n_estimators") else float(v))
            for k, v in tun.iloc[0][list(grid)].items()}
    print("[lgbm] best params:", best)
    json.dump(best, open(p("reports/tables/lgbm_best_params.json"), "w"), indent=1)

    # ---- 2. retrain at every origin with the chosen params
    fcs = []
    for o in origins:
        out, m, feats, tr = fit_predict_origin(pn, o, best, seed, h, return_model=True)
        fcs.append(out)
        print(f"[lgbm] origin {o.name}: trained on {len(tr):,} rows (targets <= week {o.train_end})")
        if o.name == "holdout":
            imp = pd.DataFrame({"feature": feats, "gain": m.booster_.feature_importance("gain")})
            imp["gain_share"] = imp.gain / imp.gain.sum()
            imp.sort_values("gain", ascending=False).to_csv(p("reports/tables/lgbm_feature_importance.csv"), index=False)
            m.booster_.save_model(str(p(f"{cfg['paths']['cache_dir']}/lgbm_holdout.txt")))
    fc = pd.concat(fcs, ignore_index=True)
    fc = add_intervals(fc, pn, [o.name for o in origins])
    fc["info"] = json.dumps(best)
    dest = ROOT / cfg["paths"]["cache_dir"] / "lgbm_forecasts.parquet"
    fc.to_parquet(dest, index=False)
    print(f"[lgbm] wrote {dest} ({len(fc)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
