"""Home page: load a portfolio (the dummy CSV or an uploaded one) and stash
it in st.session_state for the other pages. Presentation only -- every
number here comes from credit_portfolio's public APIs, no new business
logic."""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from credit_portfolio.io import load_portfolio
from credit_portfolio.valuation import portfolio_ead

DEFAULT_DATA_PATH = REPO_ROOT / "Dummy_loan_data.csv"

st.set_page_config(page_title="Credit Portfolio Modelling", layout="wide")
st.title("Credit Portfolio Modelling")

st.write(
    "Loads a portfolio of customers/facilities and makes it available to every "
    "other page in the sidebar (RWA & ECL, Loss Distribution, RAROC, Cashflow, "
    "Tranching & Waterfall)."
)

source = st.radio("Portfolio source", ["Dummy data (Dummy_loan_data.csv)", "Upload a CSV"])

csv_path = None
if source.startswith("Dummy"):
    csv_path = DEFAULT_DATA_PATH
else:
    uploaded = st.file_uploader("Loan-level CSV (same columns as Dummy_loan_data.csv)", type="csv")
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.getvalue())
        tmp.close()
        csv_path = Path(tmp.name)

if csv_path is not None and st.button("Load portfolio"):
    portfolio = load_portfolio(str(csv_path), portfolio_id="10001", portfolio_name="Portfolio")
    st.session_state["portfolio"] = portfolio
    st.success(f"Loaded portfolio with {portfolio.num_customers} customers.")

if "portfolio" in st.session_state:
    portfolio = st.session_state["portfolio"]
    num_facilities = sum(len(c.facility_list) for c in portfolio.customer_list)

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", portfolio.num_customers)
    col2.metric("Facilities", num_facilities)
    col3.metric("Total EAD", f"{portfolio_ead(portfolio):,.0f}")
else:
    st.info("Load a portfolio above to get started.")
