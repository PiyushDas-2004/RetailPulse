"""Proof that no LightGBM feature uses information from after the forecast origin.

Method: build features at origin T, then overwrite EVERY target value after T with garbage and rebuild.
If any feature changed, it must have read the future. Known-in-advance columns (calendar, SNAP schedule,
planned price) do not depend on the target, so they are unaffected by construction and also must not change.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.lgbm_features import Panel, feature_names, origin_features, training_frame


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    n_weeks = 277
    ds = pd.date_range("2011-01-29", periods=n_weeks, freq="7D")
    rows = []
    for k in range(5):
        start = 0 if k < 3 else 30
        y = rng.poisson(20 + 5 * np.sin(np.arange(n_weeks) / 8.3), n_weeks).astype(float)
        rows.append(pd.DataFrame({"unique_id": f"S{k}", "ds": ds[start:], "y": y[start:]}))
    df = pd.concat(rows)
    meta = pd.DataFrame({"unique_id": [f"S{k}" for k in range(5)], "level": ["total", "store", "store", "sku_store", "sku_store"],
                         "cat_id": "FOODS", "dept_id": "FOODS_1", "store_id": ["ALL", "CA_1", "TX_1", "CA_1", "WI_1"],
                         "state_id": ["ALL", "CA", "TX", "CA", "WI"]})
    cal = pd.DataFrame({"ds": ds, "wm_yr_wk": [11101 + i for i in range(n_weeks)], "christmas": 0, "thanksgiving": 0,
                        "easter": 0, "superbowl": 0, "n_events": 0, "snap_CA": 3, "snap_TX": 2, "snap_WI": 1})
    prices = pd.DataFrame({"unique_id": "S3", "ds": ds[30:], "price": 2.5})
    return Panel(df, meta, cal, prices, n_weeks)


@pytest.mark.parametrize("T", [60, 120, 211, 263])
def test_origin_features_ignore_the_future(panel, T):
    before = origin_features(panel, T, range(1, 14))
    original = panel.Y.copy()
    try:
        panel.Y[:, T + 1:] = 1e9  # corrupt the future
        after = origin_features(panel, T, range(1, 14))
    finally:
        panel.Y = original
    cols = feature_names(before)
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_training_frame_targets_never_after_train_end(panel):
    tr = training_frame(panel, 150, range(1, 14))
    assert tr.target_week.max() <= 150
    assert (tr.origin_T + tr.h == tr.target_week).all()


def test_training_frame_unchanged_when_future_corrupted(panel):
    a = training_frame(panel, 150, range(1, 14))
    original = panel.Y.copy()
    try:
        panel.Y[:, 151:] = -7.0
        b = training_frame(panel, 150, range(1, 14))
    finally:
        panel.Y = original
    pd.testing.assert_frame_equal(a, b)


def test_seasonal_lag_is_always_in_the_past(panel):
    f = origin_features(panel, 200, range(1, 14))
    # slag52 for h must equal y[T+h-52], which is <= T
    for h in range(1, 14):
        row = f[(f.h == h) & (f.series == 0)].iloc[0]
        assert 200 + h - 52 <= 200
        assert row.slag52 == pytest.approx(panel.Y[0, 200 + h - 52] / row.scale)
