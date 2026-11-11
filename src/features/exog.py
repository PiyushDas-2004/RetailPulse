"""Exogenous regressors that are KNOWN IN ADVANCE at forecast time (calendar, holidays, SNAP schedule, Fourier terms).

None of these use the target, so they cannot leak future demand. SNAP days are a published government schedule
(the 10 SNAP days per month per state are fixed in advance), and holidays are on the calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIOD = 365.25 / 7  # 52.18 weeks: the true annual period of weekly data (52 is an approximation)


def fourier(week_idx: np.ndarray, K: int, period: float = PERIOD) -> np.ndarray:
    t = np.asarray(week_idx, float)
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    return np.column_stack(cols) if cols else np.empty((len(t), 0))


def snap_for(state: str, cal: pd.DataFrame) -> np.ndarray:
    if state in ("CA", "TX", "WI"):
        return cal[f"snap_{state}"].to_numpy(float)
    return cal[["snap_CA", "snap_TX", "snap_WI"]].mean(1).to_numpy(float)  # national series: average schedule


def calendar_regressors(state: str, cal: pd.DataFrame) -> pd.DataFrame:
    """Per-week regressors for the full 277-week calendar (row i = global week i)."""
    return pd.DataFrame({
        "christmas": cal.christmas.to_numpy(float),
        "thanksgiving": cal.thanksgiving.to_numpy(float),
        "easter": cal.easter.to_numpy(float),
        "superbowl": cal.superbowl.to_numpy(float),
        "snap_days": snap_for(state, cal),
    })
