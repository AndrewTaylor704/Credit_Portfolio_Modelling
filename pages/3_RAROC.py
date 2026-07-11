from datetime import datetime

import streamlit as st

from streamlit_common import require_portfolio

from credit_portfolio.models import FlatFundingRate, FlatOperatingCostRate
from credit_portfolio.raroc import customer_raroc, portfolio_raroc

st.title("RAROC")
portfolio = require_portfolio()

customer_ids = [c.customerid for c in portfolio.customer_list]
selection = st.sidebar.selectbox("Scope", ["Portfolio total"] + customer_ids)

funding_rate = st.sidebar.number_input("Funding rate (annualised)", min_value=0.0, max_value=0.2, value=0.03, step=0.005)
opex_rate = st.sidebar.number_input("Operating cost rate (annualised)", min_value=0.0, max_value=0.1, value=0.005, step=0.001)
capital_basis = st.sidebar.radio("Capital basis", ["regulatory", "economic"])
tax_rate = st.sidebar.slider("Tax rate", 0.0, 0.5, 0.0, 0.01)
target_capital_ratio = st.sidebar.slider("Target capital ratio (regulatory basis)", 0.04, 0.20, 0.08, 0.01)
confidence = st.sidebar.slider("Confidence (economic basis)", 0.95, 0.9999, 0.999, 0.0001)
horizon = st.sidebar.number_input("Horizon (years)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)

funding_model = FlatFundingRate(funding_rate)
opex_model = FlatOperatingCostRate(opex_rate)

as_of_date = datetime.now()

if selection == "Portfolio total":
    result = portfolio_raroc(
        portfolio, funding_model, opex_model, as_of_date=as_of_date, horizon=horizon,
        capital_basis=capital_basis, tax_rate=tax_rate,
        target_capital_ratio=target_capital_ratio, confidence=confidence,
    )
else:
    customer = next(c for c in portfolio.customer_list if c.customerid == selection)
    result = customer_raroc(
        customer, funding_model, opex_model, as_of_date=as_of_date, horizon=horizon,
        capital_basis=capital_basis, tax_rate=tax_rate,
        target_capital_ratio=target_capital_ratio, confidence=confidence,
    )

st.metric(f"RAROC ({selection})", f"{result:.2%}")
st.caption(
    "RAROC = (income − expected loss) × (1 − tax_rate) / capital, where income nets interest, fees, "
    "funding cost, and operating cost over the horizon. Economic capital is a standalone (non-diversified) "
    "closed-form estimate, not a portfolio-diversified figure."
)
