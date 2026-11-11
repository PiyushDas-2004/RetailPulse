"""Run every per-series statistical model at every evaluation origin (4 CV folds + holdout).

Usage:  python -m src.models.run_statistical [--origins cv1,cv2,...] [--models SARIMA,ETS] [--limit N]
Output: outputs/cache/stat_forecasts.parquet (long: unique_id, origin, ds, week, h, model, mean, lo80, hi80, lo95, hi95, info)
        reports/tables/dhr_fourier_K.csv
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.load import load_calendar, load_series, make_origins, week_index
from src.features.exog import calendar_regressors
from src.models.statistical import MODELS, fit_predict, select_fourier_K
from src.utils import ROOT, load_config, p, set_seed


def _run_task(uid, y_all, wk_all, state, origin, h, models, K, cal, seed):
    import warnings
    warnings.filterwarnings("ignore")
    ctx = {"cal_reg": calendar_regressors(state, cal), "cal_ds": cal.ds, "K": K, "seed": seed, "ma_window": 4}
    tr = wk_all <= origin.train_end
    wk_f = np.arange(origin.test_start, origin.test_start + h)
    rows = []
    for name in models:
        t0 = time.time()
        try:
            o = fit_predict(name, y_all[tr], wk_all[tr], wk_f, ctx)
        except Exception as exc:  # noqa: BLE001 - record failure, fall back to seasonal naive so the table stays complete
            o = fit_predict("SeasonalNaive", y_all[tr], wk_all[tr], wk_f, ctx)
            o["info"] = f"FAILED ({type(exc).__name__}); seasonal-naive fallback"
        rows.append(pd.DataFrame({"unique_id": uid, "origin": origin.name, "week": wk_f, "h": np.arange(1, h + 1), "model": name,
                                  "mean": o["mean"], "lo80": o["lo80"], "hi80": o["hi80"], "lo95": o["lo95"], "hi95": o["hi95"],
                                  "info": o["info"], "fit_seconds": time.time() - t0}))
    return pd.concat(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", default="cv1,cv2,cv3,cv4,holdout")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    h = cfg["problem"]["horizon"]
    df, meta = load_series()
    cal = load_calendar()
    origins = {o.name: o for o in make_origins(cfg)}
    sel = [origins[o] for o in args.origins.split(",")]
    models = args.models.split(",")
    ids = meta.unique_id.tolist()[: args.limit or None]
    state = meta.set_index("unique_id").state_id.to_dict()
    groups = {uid: (g.y.to_numpy(float), week_index(g.ds).to_numpy()) for uid, g in df.groupby("unique_id")}

    # Fourier order K per series: chosen ONCE by AICc on data before the first CV test window (no evaluation data seen)
    k_path = p("reports/tables/dhr_fourier_K.csv")
    if "DHR-ARIMA" in models:
        if k_path.exists():
            kdf = pd.read_csv(k_path)
        else:
            cut = origins["cv1"].train_end

            def _k(uid):
                y, wk = groups[uid]
                m = wk <= cut
                K, sc = select_fourier_K(y[m], wk[m], calendar_regressors(state[uid], cal), tuple(cfg["models"]["dhr_fourier_K"]))
                return {"unique_id": uid, "K": K, **{f"aicc_K{k}": v for k, v in sc.items()}}
            t0 = time.time()
            kdf = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(delayed(_k)(u) for u in meta.unique_id))
            kdf.to_csv(k_path, index=False)
            print(f"[stat] Fourier K selection done in {time.time()-t0:.0f}s: {kdf.K.value_counts().to_dict()}")
        Kmap = kdf.set_index("unique_id").K.to_dict()
    else:
        Kmap = {}

    t0 = time.time()
    tasks = [(uid, o) for o in sel for uid in ids]
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_run_task)(uid, *groups[uid], state[uid], o, h, models, Kmap.get(uid, 6), cal, cfg["seed"]) for uid, o in tasks)
    out = pd.concat(res, ignore_index=True)
    dest = ROOT / cfg["paths"]["cache_dir"] / "stat_forecasts.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and args.limit == 0:
        old = pd.read_parquet(dest)
        old = old[~(old.model.isin(models) & old.origin.isin([o.name for o in sel]))]
        out = pd.concat([old, out], ignore_index=True)
    out.to_parquet(dest, index=False)
    print(f"[stat] {len(tasks)} tasks in {time.time()-t0:.0f}s -> {dest}")
    print(out.groupby("model").fit_seconds.sum().round(0))
    print("failures:", out[out["info"].str.startswith("FAILED")].groupby("model").unique_id.nunique().to_dict())


if __name__ == "__main__":
    main()
