from datetime import datetime

import pandas as pd
import streamlit as st

from streamlit_common import require_portfolio

from credit_portfolio.valuation import customer_ead, customer_ecl, customer_rwa, rwa_standardised

st.title("RWA & ECL")
portfolio = require_portfolio()

as_of_date = st.sidebar.date_input("As of date", value=datetime.now().date())
as_of_datetime = datetime.combine(as_of_date, datetime.min.time())

approach = st.sidebar.radio("Approach", ["IRB", "Standardised"])
jurisdiction = None
if approach == "Standardised":
    jurisdiction = st.sidebar.selectbox("Jurisdiction", ["BCBS", "EU", "UK"])

rows = []
for customer in portfolio.customer_list:
    ead = customer_ead(customer, as_of_datetime)
    ecl = customer_ecl(customer, as_of_datetime)
    if approach == "IRB":
        rwa = customer_rwa(customer, as_of_datetime)
    else:
        rwa = sum(rwa_standardised(f, as_of_datetime, jurisdiction=jurisdiction) for f in customer.facility_list)
    rows.append({"Customer": customer.customerid, "EAD": ead, "ECL": ecl, "RWA": rwa})

df = pd.DataFrame(rows)

col1, col2, col3 = st.columns(3)
col1.metric("Total EAD", f"{df['EAD'].sum():,.0f}")
col2.metric("Total ECL", f"{df['ECL'].sum():,.0f}")
col3.metric("Total RWA", f"{df['RWA'].sum():,.0f}")

st.dataframe(df.style.format({"EAD": "{:,.0f}", "ECL": "{:,.0f}", "RWA": "{:,.0f}"}), use_container_width=True)
