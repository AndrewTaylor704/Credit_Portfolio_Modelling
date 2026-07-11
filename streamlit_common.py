"""Shared bootstrap/helpers for the Streamlit pages -- UI glue only, no
business logic (that all lives in credit_portfolio)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from credit_portfolio.domain import Portfolio


def require_portfolio() -> Portfolio:
    if "portfolio" not in st.session_state:
        st.warning("Load a portfolio on the Home page first.")
        st.stop()
    return st.session_state["portfolio"]
