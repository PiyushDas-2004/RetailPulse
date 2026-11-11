import numpy as np
import pytest

from src.inventory.policy import order_up_to, reorder_point, safety_stock, simulate_lost_sales, z_value


def test_z_values_match_normal_table():
    assert z_value(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert z_value(0.90) == pytest.approx(1.2816, abs=1e-4)
    assert z_value(0.50) == pytest.approx(0.0, abs=1e-12)


def test_safety_stock_formula():
    # SS = z * sigma * sqrt(L + R)
    assert safety_stock(10, 4, 0.95) == pytest.approx(1.6449 * 10 * 2, abs=1e-3)
    assert safety_stock(10, 3, 0.95, review_period=1) == pytest.approx(1.6449 * 10 * 2, abs=1e-3)


def test_safety_stock_grows_with_service_level_and_lead_time():
    assert safety_stock(5, 2, 0.98) > safety_stock(5, 2, 0.95) > safety_stock(5, 2, 0.90)
    assert safety_stock(5, 4, 0.95) > safety_stock(5, 1, 0.95)


def test_reorder_point_and_order_up_to():
    assert reorder_point(100, 10, 4, 0.95) == pytest.approx(100 + 1.6449 * 20, abs=1e-3)
    assert order_up_to(125, 10, 4, 1, 0.95) == pytest.approx(125 + z_value(0.95) * 10 * np.sqrt(5), abs=1e-9)


def test_simulation_perfect_information_has_no_stockouts():
    d = np.full(10, 20.0)
    r = simulate_lost_sales(d, S=np.full(10, 60.0), L=2, initial_on_hand=20.0, initial_pipeline=np.array([20.0, 20.0]))
    assert r["stockout_weeks"] == 0 and r["fill_rate"] == 1.0


def test_simulation_zero_stock_loses_all_sales():
    d = np.full(5, 10.0)
    r = simulate_lost_sales(d, S=np.zeros(5), L=1, initial_on_hand=0.0, initial_pipeline=np.zeros(1))
    assert r["fill_rate"] == 0.0 and r["lost_units"] == 50.0 and r["shortage_cost"] == 450.0


def test_simulation_conserves_units():
    rng = np.random.default_rng(0)
    d = rng.poisson(15, 30).astype(float)
    r = simulate_lost_sales(d, S=np.full(30, 50.0), L=2, initial_on_hand=30.0, initial_pipeline=np.array([15.0, 15.0]))
    assert r["fill_rate"] * r["demand"] + r["lost_units"] == pytest.approx(r["demand"])
