import pandas as pd
import streamlit as st

from streamlit_common import require_portfolio

from credit_portfolio.risk import simulate_cashflow_paths
from credit_portfolio.securitisation import Tranche, run_waterfall

st.title("Tranching & Cash Waterfall")
portfolio = require_portfolio()

st.caption(
    "Pro-rata pass-through waterfall: each period, cumulative losses erode tranche notional "
    "in reverse-seniority order (equity first), and each tranche receives a share of that "
    "period's actual cash proportional to its surviving notional. Not a full sequential-pay "
    "structure with independent per-tranche coupons."
)

correlation = st.sidebar.slider("Systemic correlation", 0.0, 0.9, 0.15, 0.01)
num_sims = st.sidebar.select_slider("Number of simulations", options=[100, 500, 1_000, 5_000], value=500)
periods_per_year = st.sidebar.selectbox("Periods per year", [4, 12, 1], index=0)
horizon = st.sidebar.number_input("Deal horizon (years)", min_value=1.0, max_value=10.0, value=3.0, step=1.0)

boundaries = st.sidebar.slider("Tranche boundaries (equity | mezzanine | senior)", 0.0, 1.0, (0.05, 0.15), 0.01)
tranches = [
    Tranche("equity", 0.0, boundaries[0]),
    Tranche("mezzanine", boundaries[0], boundaries[1]),
    Tranche("senior", boundaries[1], 1.0),
]

if st.button("Run simulation"):
    with st.spinner("Simulating multi-period default paths..."):
        result = simulate_cashflow_paths(
            portfolio, correlation=correlation, num_sims=num_sims, periods_per_year=periods_per_year, horizon=horizon
        )
        waterfall = run_waterfall(tranches, result)
    st.session_state["cashflow_paths_result"] = result
    st.session_state["waterfall_result"] = waterfall
    st.session_state["waterfall_tranches"] = tranches

if "waterfall_result" in st.session_state:
    result = st.session_state["cashflow_paths_result"]
    waterfall = st.session_state["waterfall_result"]
    tranches = st.session_state["waterfall_tranches"]

    period_index = list(range(len(result.period_dates)))

    summary_rows = []
    cash_by_period = pd.DataFrame(index=period_index)
    notional_by_period = pd.DataFrame(index=period_index)
    for tranche in tranches:
        tc = waterfall[tranche.name]
        notional = tranche.thickness * result.total_notional
        total_cash = tc.cash.sum(axis=1).mean()
        summary_rows.append({
            "Tranche": tranche.name,
            "Notional": notional,
            "Avg total cash": total_cash,
            "Cash yield": total_cash / notional if notional else float("nan"),
        })
        cash_by_period[tranche.name] = tc.cash.mean(axis=0)
        notional_by_period[tranche.name] = tc.remaining_notional.mean(axis=0)

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df.style.format({"Notional": "{:,.0f}", "Avg total cash": "{:,.0f}", "Cash yield": "{:.2%}"}),
        use_container_width=True,
    )

    st.subheader("Average cash received per period, by tranche")
    st.line_chart(cash_by_period)

    st.subheader("Average remaining notional per period, by tranche")
    st.line_chart(notional_by_period)
else:
    st.info("Set parameters in the sidebar and click 'Run simulation'.")
