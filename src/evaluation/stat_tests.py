"""Statistical comparison of forecast accuracy.

* Diebold-Mariano test (1995) with the Harvey-Leybourne-Newbold (1997) small-sample correction.
  H0: equal expected loss. Loss = absolute error. The loss differential of multi-step forecasts is autocorrelated,
  so its variance uses autocovariances up to lag h-1 (Newey-West style, rectangular kernel as in DM).
* Paired bootstrap across series of the mean MASE difference (series are the resampling unit, which respects that
  errors within a series are dependent).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(e1, e2, h: int = 1, power: int = 1) -> tuple[float, float]:
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var)
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN correction
    stat = dm * k
    pval = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(pval)


def paired_bootstrap(a, b, n_boot: int = 5000, seed: int = 42) -> dict:
    """CI for mean(a - b) by resampling series. a, b: per-series metric (e.g. mean MASE) for two models."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi), "n_series": int(ok.sum()),
            "share_a_better": float((d < 0).mean()), "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if len(d) > 10 else np.nan}
