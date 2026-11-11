"""RetailPulse dashboard.   Run:  streamlit run app/streamlit_app.py   (after `make all`)

Reads only files produced by the pipeline: outputs/forecasts.csv, outputs/powerbi/*.csv, reports/tables/*.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Naive": "#9a9994", "SeasonalNaive": "#52514e", "MovingAverage": "#c3c2b7", "SES": "#e87ba4", "ETS": "#1baf7a",
          "SARIMA": "#2a78d6", "DHR-ARIMA": "#6250d6", "Prophet": "#eda100", "LightGBM": "#eb6834", "Ensemble": "#e34948"}

st.set_page_config(page_title="RetailPulse", layout="wide")


@st.cache_data
def load():
    pb = ROOT / "outputs/powerbi"
    d = {
        "actuals": pd.read_csv(pb / "fact_actuals.csv", parse_dates=["ds"]),
        "future": pd.read_csv(ROOT / "outputs/forecasts.csv", parse_dates=["ds"]),
        "backtest": pd.read_csv(pb / "fact_backtest_forecasts.csv", parse_dates=["ds"])
        if (pb / "fact_backtest_forecasts.csv").exists() else None,
        "series": pd.read_csv(pb / "dim_series.csv"),
        "lb_level": pd.read_csv(ROOT / "reports/tables/leaderboard_cv_by_level.csv"),
        "lb_class": pd.read_csv(ROOT / "reports/tables/leaderboard_cv_by_sku_class.csv"),
        "lb_cv": pd.read_csv(ROOT / "reports/tables/leaderboard_cv_overall.csv"),
        "lb_ho": pd.read_csv(ROOT / "reports/tables/leaderboard_holdout_overall.csv"),
        "recs": pd.read_csv(ROOT / "outputs/inventory_recommendations.csv"),
        "sim": pd.read_csv(ROOT / "reports/tables/inventory_simulation_summary.csv"),
    }
    return d


d = load()
st.title("RetailPulse: demand forecasting & inventory (M5 Walmart public data)")
st.caption("Weekly unit demand, 211 series. Backtests: 4 rolling-origin CV folds + 13-week holdout. "
           "Inventory figures are SIMULATED under stated assumptions; this is public data, not any retailer's own.")

with st.sidebar:
    st.header("Series")
    levels = ["total", "category", "department", "store", "store_dept", "sku_store"]
    lvl = st.selectbox("Hierarchy level", levels, index=0)
    ids = d["series"][d["series"].level == lvl].unique_id.sort_values().tolist()
    uid = st.selectbox("Series", ids)
    info = d["series"].set_index("unique_id").loc[uid]
    st.markdown(f"**Demand class:** {info.sbc_class}  \n**Seasonal strength:** {info.seasonal_strength:.2f}  \n"
                f"**Trend strength:** {info.trend_strength:.2f}  \n**Zero weeks:** {100*info.share_zeros:.0f}%  \n"
                f"**Mean units/week:** {info['mean']:,.1f}")
    fut_models = sorted(d["future"].model.unique())
    rec_model = d["future"][(d["future"].unique_id == uid) & d["future"].recommended].model.iloc[0]
    models = st.multiselect("Models to show", fut_models, default=[rec_model])

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Backtest", "Model comparison", "Inventory"])

with tab1:
    st.subheader(f"{uid}: history and 13-week forecast (recommended model: {rec_model})")
    hist = d["actuals"][d["actuals"].unique_id == uid].tail(104)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.ds, y=hist.actual_units, name="actual", line=dict(color="#52514e", width=1.5)))
    for m in models:
        f = d["future"][(d["future"].unique_id == uid) & (d["future"].model == m)].sort_values("ds")
        c = COLORS.get(m, "#2a78d6")
        if f.lo95.notna().any():
            fig.add_trace(go.Scatter(x=list(f.ds) + list(f.ds[::-1]), y=list(f.hi95) + list(f.lo95[::-1]), fill="toself",
                                     fillcolor=c, opacity=0.12, line=dict(width=0), name=f"{m} 95%", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=list(f.ds) + list(f.ds[::-1]), y=list(f.hi80) + list(f.lo80[::-1]), fill="toself",
                                     fillcolor=c, opacity=0.25, line=dict(width=0), name=f"{m} 80%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=f.ds, y=f["forecast_units"] if "forecast_units" in f else f["mean"], name=m,
                                 line=dict(color=c, width=2)))
    fig.update_layout(height=430, hovermode="x unified", yaxis_title="units / week", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    f = d["future"][(d["future"].unique_id == uid) & (d["future"].model == rec_model)].sort_values("ds")
    st.dataframe(f[["ds", "h", "mean", "lo80", "hi80", "lo95", "hi95"]].round(1), hide_index=True, width="stretch")

with tab2:
    st.subheader("Backtest: forecasts made at each origin vs what actually happened")
if d["backtest"] is None:
    tab2.info("Run `make export` to create outputs/powerbi/fact_backtest_forecasts.csv (kept out of git, 12 MB).")
else:
  with tab2:
    bt = d["backtest"][d["backtest"].unique_id == uid]
    org = st.radio("Origin", sorted(bt.origin.unique(), key=lambda x: (x == "holdout", x)), horizontal=True, index=4)
    b = bt[bt.origin == org]
    fig = go.Figure()
    ctx = d["actuals"][(d["actuals"].unique_id == uid) & (d["actuals"].ds <= b.ds.max()) & (d["actuals"].ds >= b.ds.min() - pd.Timedelta(weeks=26))]
    fig.add_trace(go.Scatter(x=ctx.ds, y=ctx.actual_units, name="actual", line=dict(color="#52514e", width=1.5)))
    for m in sorted(b.model.unique()):
        x = b[b.model == m].sort_values("ds")
        fig.add_trace(go.Scatter(x=x.ds, y=x.forecast_units, name=m, line=dict(color=COLORS.get(m), width=1.6),
                                 visible=True if m in models + ["SeasonalNaive"] else "legendonly"))
    fig.update_layout(height=400, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    sc = b.groupby("model").apply(lambda g: pd.Series({"WAPE": g.abs_error.sum() / g.actual_units.sum(),
                                                       "Bias": g.error.sum() / g.actual_units.sum()}), include_groups=False)
    st.dataframe(sc.sort_values("WAPE").round(3), width="stretch")

with tab3:
    st.subheader("Leaderboards (lower MASE/WAPE is better; MASE < 1 beats in-sample seasonal naive)")
    c1, c2 = st.columns(2)
    c1.markdown("**CV folds (model selection)**")
    c1.dataframe(d["lb_cv"].round(3), hide_index=True, width="stretch")
    c2.markdown("**Holdout (final, untouched)**")
    c2.dataframe(d["lb_ho"].round(3), hide_index=True, width="stretch")
    st.markdown("**Mean CV MASE by hierarchy level**")
    piv = d["lb_level"].pivot(index="model", columns="level", values="mean_MASE")
    st.dataframe(piv.round(3).style.highlight_min(axis=0, color="#cfe3fa"), width="stretch")
    st.markdown("**Mean CV MASE by demand class**")
    piv2 = d["lb_class"].pivot(index="model", columns="sku_class", values="mean_MASE")
    st.dataframe(piv2.round(3).style.highlight_min(axis=0, color="#cfe3fa"), width="stretch")

with tab4:
    st.subheader("Replenishment recommendation (lead time and service level are assumptions)")
    r = d["recs"][d["recs"].unique_id == uid]
    if len(r):
        r = r.iloc[0]
        a, b_, c, e = st.columns(4)
        a.metric("Safety stock", f"{r.safety_stock:,.0f} units")
        b_.metric("Reorder point", f"{r.reorder_point:,.0f} units")
        c.metric("Order-up-to level", f"{r.order_up_to_level:,.0f} units")
        e.metric("Risk flag", r.risk)
        st.caption(f"Model {r.forecast_model}; lead time {int(r.lead_time_weeks)} wk; service level {r.service_level:.0%}; "
                   f"sigma (CV forecast error, h=1-4) = {r.sigma_forecast_error:,.1f}; simulated holdout fill rate {r.sim_fill_rate:.1%}.")
    else:
        st.info("Inventory recommendations are computed for store x department and SKU x store series (the replenishment units).")
    st.markdown("**At-risk list** (simulated holdout + CV bias)")
    risk = d["recs"][d["recs"].risk != "ok"].sort_values("risk_score", ascending=False)
    st.dataframe(risk[["unique_id", "risk", "sbc_class", "forecast_model", "cv_bias", "sim_fill_rate", "sim_weeks_of_cover",
                       "safety_stock", "reorder_point"]].round(3), hide_index=True, width="stretch")
    st.markdown("**Simulated policy comparison** (SKU level; sum over 120 SKUs)")
    grp = st.radio("Group", ["SKU", "store_dept"], horizontal=True)
    sl = st.select_slider("Service level", sorted(d["sim"].service_level.unique()), value=0.95)
    s = d["sim"][(d["sim"].group == grp) & (d["sim"].service_level == sl)]
    fig = go.Figure()
    for pol in s.policy.unique():
        x = s[s.policy == pol].sort_values("lead_time")
        fig.add_trace(go.Scatter(x=x.lead_time, y=x.total_cost, mode="lines+markers", name=pol,
                                 line=dict(color=COLORS.get(pol, "#9a9994"))))
    fig.update_layout(height=360, xaxis_title="lead time (weeks)", yaxis_title="total cost proxy (holding + shortage)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(s[["policy", "lead_time", "fill_rate", "stockout_rate", "avg_on_hand", "total_cost", "total_cost_vs_SeasonalNaive_pct"]].round(3),
                 hide_index=True, width="stretch")
