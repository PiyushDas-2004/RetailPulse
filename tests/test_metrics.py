import numpy as np
import pytest

from src.evaluation.metrics import bias, coverage, mase, mase_scale, rmse, smape, wape
from src.evaluation.stat_tests import dm_test


def test_wape_hand_computed():
    y = np.array([10, 0, 30])
    f = np.array([12, 1, 27])
    assert wape(y, f) == pytest.approx((2 + 1 + 3) / 40)


def test_wape_defined_with_zero_actuals_and_nan_when_all_zero():
    assert np.isfinite(wape([0, 5], [1, 5]))
    assert np.isnan(wape([0, 0], [1, 1]))


def test_bias_sign_convention():
    assert bias([10, 10], [12, 12]) == pytest.approx(0.2)   # over-forecast -> positive
    assert bias([10, 10], [8, 8]) == pytest.approx(-0.2)    # under-forecast -> negative


def test_rmse():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_mase_scale_is_seasonal_naive_in_sample_mae():
    y = np.arange(60, dtype=float)  # y[t] - y[t-52] = 52 everywhere
    assert mase_scale(y, 52) == pytest.approx(52.0)
    assert mase([1, 2], [1 + 52, 2 + 52], mase_scale(y, 52)) == pytest.approx(1.0)


def test_mase_scale_falls_back_to_lag1_when_seasonal_scale_is_zero():
    y = np.tile([1.0, 2.0], 60)  # period-2 pattern: lag-52 differences are 0
    assert mase_scale(y, 52) == pytest.approx(1.0)


def test_smape_handles_double_zero():
    assert smape([0, 10], [0, 10]) == 0.0


def test_coverage():
    assert coverage([1, 2, 3, 4], [0, 0, 0, 5], [5, 5, 5, 6]) == 0.75


def test_dm_identical_forecasts_not_significant():
    rng = np.random.default_rng(0)
    e = rng.normal(size=60)
    stat, p = dm_test(e, e.copy(), h=4)
    assert p == 1.0 or abs(stat) < 1e-9


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    e1 = rng.normal(0, 1, 200)
    e2 = rng.normal(0, 3, 200)
    stat, p = dm_test(e1, e2, h=1)
    assert stat < 0 and p < 0.01
