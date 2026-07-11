import matplotlib.pyplot as plt
import streamlit as st

from streamlit_common import require_portfolio

from credit_portfolio.risk import loss_dist_quantile, simulate_loss_distribution

st.title("Loss Distribution (Monte Carlo)")
portfolio = require_portfolio()

correlation = st.sidebar.slider("Systemic correlation", 0.0, 0.9, 0.15, 0.01)
num_sims = st.sidebar.select_slider("Number of simulations", options=[1_000, 5_000, 10_000, 50_000, 100_000], value=10_000)
num_bins = st.sidebar.select_slider("Number of bins", options=[100, 500, 1_000, 5_000], value=1_000)
horizon = st.sidebar.number_input("Horizon (years)", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
confidence = st.sidebar.slider("Quantile confidence", 0.90, 0.999, 0.999, 0.001)

if st.button("Run simulation"):
    with st.spinner("Simulating..."):
        result = simulate_loss_distribution(
            portfolio, correlation=correlation, num_sims=num_sims, num_bins=num_bins, horizon=horizon
        )
    st.session_state["mc_result"] = result

if "mc_result" in st.session_state:
    result = st.session_state["mc_result"]
    quantile = loss_dist_quantile(result.loss_dist, confidence)

    col1, col2 = st.columns(2)
    col1.metric("Total notional", f"{result.total_notional:,.0f}")
    col2.metric(f"{confidence:.1%} loss quantile", f"{quantile:,.0f}" if quantile is not False else "n/a")

    fig, ax = plt.subplots()
    ax.plot(result.loss_dist[:, 1], result.loss_dist[:, 0])
    ax.set_xlabel("Loss amount")
    ax.set_ylabel("P(loss >= threshold)")
    ax.set_title("Portfolio loss distribution (survival function)")
    ax.set_xlim(0, result.loss_dist[:, 1].max() / 5)
    st.pyplot(fig)
else:
    st.info("Set parameters in the sidebar and click 'Run simulation'.")
