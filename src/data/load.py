"""Load the processed weekly dataset and define the time splits (single source of truth)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils import ROOT, load_config


def load_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    d = ROOT / cfg["paths"]["processed_dir"]
    df = pd.read_parquet(d / "weekly_series.parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    meta = pd.read_csv(d / "series_meta.csv")
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True), meta


def load_calendar() -> pd.DataFrame:
    cfg = load_config()
    cal = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "weekly_calendar.csv", parse_dates=["ds"])
    return cal


def load_prices() -> pd.DataFrame:
    cfg = load_config()
    pr = pd.read_parquet(ROOT / cfg["paths"]["processed_dir"] / "sku_prices.parquet")
    pr["ds"] = pd.to_datetime(pr["ds"])
    return pr


@dataclass(frozen=True)
class Origin:
    name: str          # e.g. "tune1", "cv1", "holdout"
    kind: str          # "tuning" | "cv" | "holdout"
    train_end: int     # last training week index (inclusive)
    test_start: int
    test_end: int      # inclusive


def make_origins(cfg: dict | None = None) -> list[Origin]:
    """Expanding-window origins, ordered in time. Week indices are 0..n_full_weeks-1 on the global calendar."""
    cfg = cfg or load_config()
    n = cfg["data"]["n_full_weeks"]
    h = cfg["problem"]["horizon"]
    step = cfg["problem"]["cv_step"]
    k = cfg["problem"]["n_cv_folds"]
    t = cfg["problem"]["n_tuning_origins"]
    hold_start = n - h
    out = []
    for i in range(t + k, 0, -1):
        start = hold_start - i * step
        kind = "tuning" if i > k else "cv"
        idx = (t + k - i + 1) if kind == "tuning" else (k - i + 1)
        out.append(Origin(f"{'tune' if kind == 'tuning' else 'cv'}{idx}", kind, start - 1, start, start + h - 1))
    out.append(Origin("holdout", "holdout", hold_start - 1, hold_start, n - 1))
    return out


def week_index(ds: pd.Series | pd.DatetimeIndex, first: str = "2011-01-29") -> pd.Series:
    return ((pd.to_datetime(ds) - pd.Timestamp(first)).dt.days // 7) if isinstance(ds, pd.Series) else \
        ((pd.to_datetime(ds) - pd.Timestamp(first)).days // 7)
