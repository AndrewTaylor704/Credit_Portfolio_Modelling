from datetime import datetime

import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

from streamlit_common import require_portfolio

from credit_portfolio.cashflow import project_customer_cashflows

st.title("Cashflow Projection")
portfolio = require_portfolio()

customer_ids = [c.customerid for c in portfolio.customer_list]
selected_id = st.sidebar.selectbox("Customer", customer_ids)
customer = next(c for c in portfolio.customer_list if c.customerid == selected_id)

as_of_date = datetime.now()

enable_default = st.sidebar.checkbox("Simulate a customer-level default (cross-default)")
default_date = None
recovery_lag = relativedelta(years=1)
if enable_default:
    default_input = st.sidebar.date_input("Default date", value=(as_of_date + relativedelta(years=1)).date())
    default_date = datetime.combine(default_input, datetime.min.time())
    recovery_lag_months = st.sidebar.slider("Recovery lag (months)", 1, 36, 12)
    recovery_lag = relativedelta(months=recovery_lag_months)

schedule = project_customer_cashflows(customer, as_of_date, default_date=default_date, recovery_lag=recovery_lag)

rows = [
    {
        "Date": e.date,
        "Interest": e.interest,
        "Commitment fee": e.commitment_fee,
        "Upfront fee": e.upfront_fee,
        "Principal": e.principal,
        "Recovery": e.recovery,
        "Total": e.total,
    }
    for e in schedule.events
]
df = pd.DataFrame(rows)

if df.empty:
    st.info("No cashflow events for this customer over the selected scenario.")
else:
    st.line_chart(df.set_index("Date")[["Interest", "Commitment fee", "Upfront fee", "Principal", "Recovery"]])
    st.dataframe(
        df.style.format({c: "{:,.2f}" for c in df.columns if c != "Date"}),
        use_container_width=True,
    )
    st.metric("Total cash over schedule", f"{df['Total'].sum():,.2f}")
