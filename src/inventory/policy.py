"""Inventory policy formulas and a lost-sales periodic-review simulator.

Policy: periodic review (every R weeks) order-up-to level S, lead time L weeks.
  protection interval  P = L + R            (an order placed now must cover demand until the NEXT order arrives)
  safety stock         SS = z * sigma * sqrt(P)
  order-up-to level    S  = (forecast demand over the next P weeks) + SS
  reorder point        ROP = (forecast demand over the lead time L) + z * sigma * sqrt(L)   (continuous-review view)
  z = Phi^{-1}(cycle service level)

sigma = standard deviation of the weekly FORECAST ERROR (from cross-validation residuals), not of demand itself:
the forecast already explains the predictable part (trend, seasonality, events); only the unexplained part needs a
buffer. Using raw demand variability would double-count seasonality as "risk" and over-stock.
The sqrt(P) rule assumes weekly errors are independent with equal variance; multi-step errors are in practice
positively correlated, so this is a known (optimistic) simplification that the simulation puts to the test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def z_value(service_level: float) -> float:
    return float(stats.norm.ppf(service_level))


def safety_stock(sigma: float, lead_time: float, service_level: float, review_period: float = 0.0) -> float:
    return z_value(service_level) * sigma * np.sqrt(lead_time + review_period)


def reorder_point(forecast_lead_time_demand: float, sigma: float, lead_time: float, service_level: float) -> float:
    return forecast_lead_time_demand + safety_stock(sigma, lead_time, service_level)


def order_up_to(forecast_protection_demand: float, sigma: float, lead_time: float, review_period: float,
                service_level: float) -> float:
    return forecast_protection_demand + safety_stock(sigma, lead_time, service_level, review_period)


def simulate_lost_sales(demand: np.ndarray, S: np.ndarray, L: int, initial_on_hand: float, initial_pipeline: np.ndarray,
                        holding_cost: float = 1.0, shortage_cost: float = 9.0) -> dict:
    """Weekly lost-sales simulation of an order-up-to policy (R = 1).

    Timeline in week t: (1) receive the order placed L weeks ago, (2) review: order q_t = max(0, S_t - IP_t) where
    IP = on-hand + on-order, (3) demand d_t occurs; unmet demand is lost (retail customers walk away),
    (4) holding cost on end-of-week on-hand. With L = 0 an order arrives immediately (before demand).
    initial_pipeline[k] = quantity arriving at the start of week k (k < L), already ordered before the simulation.
    """
    T = len(demand)
    on_hand = float(initial_on_hand)
    pipeline = list(np.asarray(initial_pipeline, float)[:L]) + [0.0] * (T + L + 1)
    lost = np.zeros(T)
    sold = np.zeros(T)
    end_inv = np.zeros(T)
    orders = np.zeros(T)
    for t in range(T):
        on_hand += pipeline[t]
        ip = on_hand + sum(pipeline[t + 1: t + L + 1]) if L > 0 else on_hand
        q = max(0.0, S[t] - ip)
        orders[t] = q
        if L == 0:
            on_hand += q
        else:
            pipeline[t + L] += q
        sold[t] = min(on_hand, demand[t])
        lost[t] = demand[t] - sold[t]
        on_hand -= sold[t]
        end_inv[t] = on_hand
    total_d = demand.sum()
    return {
        "stockout_weeks": int((lost > 1e-9).sum()), "stockout_rate": float((lost > 1e-9).mean()),
        "fill_rate": float(sold.sum() / total_d) if total_d > 0 else 1.0, "lost_units": float(lost.sum()),
        "avg_on_hand": float(end_inv.mean()), "holding_cost": float(holding_cost * end_inv.sum()),
        "shortage_cost": float(shortage_cost * lost.sum()), "total_cost": float(holding_cost * end_inv.sum() + shortage_cost * lost.sum()),
        "units_ordered": float(orders.sum()), "demand": float(total_d),
    }
