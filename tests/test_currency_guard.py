from datetime import datetime

import pytest

from credit_portfolio.domain import Portfolio
from credit_portfolio.raroc import customer_raroc
from credit_portfolio.valuation import customer_ead, portfolio_ead
from tests.conftest import make_customer, make_facility


def test_customer_ead_raises_on_mixed_currency_facilities():
    customer = make_customer()
    make_facility(customer=customer, facid="F1", currency="GBP")
    make_facility(customer=customer, facid="F2", currency="USD")

    with pytest.raises(ValueError, match="different currencies"):
        customer_ead(customer)


def test_customer_ead_is_fine_with_consistent_currency():
    customer = make_customer()
    make_facility(customer=customer, facid="F1", currency="GBP")
    make_facility(customer=customer, facid="F2", currency="GBP")

    assert customer_ead(customer) > 0


def test_portfolio_ead_raises_when_customers_use_different_currencies():
    customer_gbp = make_customer(customerid="C1")
    make_facility(customer=customer_gbp, currency="GBP")
    customer_usd = make_customer(customerid="C2")
    make_facility(customer=customer_usd, currency="USD")

    portfolio = Portfolio("P1", "mixed currency")
    portfolio.add_customer(customer_gbp)
    portfolio.add_customer(customer_usd)

    with pytest.raises(ValueError, match="different currencies"):
        portfolio_ead(portfolio)


def test_raroc_also_raises_on_mixed_currency():
    from credit_portfolio.models import FlatFundingRate, FlatOperatingCostRate

    customer = make_customer()
    make_facility(customer=customer, facid="F1", currency="GBP")
    make_facility(customer=customer, facid="F2", currency="USD")

    with pytest.raises(ValueError, match="different currencies"):
        customer_raroc(customer, FlatFundingRate(0.03), FlatOperatingCostRate(0.005), as_of_date=datetime.now())
