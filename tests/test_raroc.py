from datetime import datetime

import numpy as np
import pytest
from scipy.stats import norm

from credit_portfolio.amortisation import AmortisationSchedule, Frequency
from credit_portfolio.cashflow import project_facility_cashflows
from credit_portfolio.models import (
    ExposureClass,
    FlatFundingRate,
    FlatOperatingCostRate,
    MacroConditionedPD,
)
from credit_portfolio.models.pd import asset_correlation
from credit_portfolio.raroc import customer_raroc, economic_capital, facility_raroc, portfolio_raroc, regulatory_capital
from credit_portfolio.valuation import ead, ecl
from credit_portfolio.valuation.rwa_retail import retail_mortgage_correlation
from tests.conftest import make_customer, make_facility

START = datetime(2020, 1, 1)
MATURITY = datetime(2025, 1, 1)  # 5 years


def wholesale_facility(**overrides):
    defaults = dict(
        type="Loan",
        start_date=START,
        maturity_date=MATURITY,
        limit=1_000_000.0,
        drawn_balance=1_000_000.0,
        margin=0.03,
        fee=0.01,
        lgd=0.45,
        amort_schedule=AmortisationSchedule.straight_line(Frequency.QUARTERLY),
    )
    defaults.update(overrides)
    return make_facility(**defaults)


FUNDING = FlatFundingRate(0.01)
OPEX = FlatOperatingCostRate(0.002)


def test_facility_raroc_regulatory_matches_hand_computation():
    facility = wholesale_facility()
    as_of = START
    horizon_end = datetime(2021, 1, 1)

    cashflows = project_facility_cashflows(facility, as_of, funding_model=FUNDING, opex_model=OPEX)
    expected_income = sum(
        e.interest + e.commitment_fee + e.upfront_fee - e.funding_cost - e.operating_cost
        for e in cashflows.events
        if as_of <= e.date < horizon_end
    )
    expected_el = ecl(facility, as_of)
    expected_capital = regulatory_capital(facility, as_of, target_capital_ratio=0.08)
    expected = (expected_income - expected_el) / expected_capital

    result = facility_raroc(facility, FUNDING, OPEX, as_of_date=as_of, target_capital_ratio=0.08)
    assert result == pytest.approx(expected)


def test_facility_raroc_tax_rate_scales_the_numerator():
    facility = wholesale_facility()
    pretax = facility_raroc(facility, FUNDING, OPEX, as_of_date=START, tax_rate=0.0)
    posttax = facility_raroc(facility, FUNDING, OPEX, as_of_date=START, tax_rate=0.25)
    assert posttax == pytest.approx(pretax * 0.75)


def test_economic_capital_matches_hand_computation_for_corporate():
    facility = wholesale_facility()
    as_of = START
    pd = facility.customer.probdef
    lgd = facility.lgd
    R = asset_correlation(pd, facility.customer.turnover)
    wcdr = norm.cdf((norm.ppf(pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R))
    expected = ead(facility, as_of) * lgd * (wcdr - pd)

    assert economic_capital(facility, as_of, confidence=0.999) == pytest.approx(expected)


def test_economic_capital_uses_retail_mortgage_correlation_not_wholesale():
    customer = make_customer(exposure_class=ExposureClass.RETAIL_MORTGAGE, probdef=0.02, turnover=0)
    facility = wholesale_facility(customer=customer, lgd=0.2)
    as_of = START

    pd = customer.probdef
    lgd = facility.lgd
    R = retail_mortgage_correlation()
    wcdr = norm.cdf((norm.ppf(pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R))
    expected = ead(facility, as_of) * lgd * (wcdr - pd)

    assert economic_capital(facility, as_of, confidence=0.999) == pytest.approx(expected)

    # Sanity: using the wholesale correlation would give a different answer.
    wholesale_R = asset_correlation(pd, customer.turnover)
    assert wholesale_R != pytest.approx(R)


def test_economic_capital_raises_for_specialised_lending():
    customer = make_customer(exposure_class=ExposureClass.SPECIALISED_LENDING)
    facility = wholesale_facility(customer=customer, supervisory_slot="Good")
    with pytest.raises(NotImplementedError):
        economic_capital(facility, START)


def test_economic_capital_rejects_non_ttc_pd_model():
    facility = wholesale_facility()
    with pytest.raises(ValueError, match="through-the-cycle"):
        economic_capital(facility, START, pd_model=MacroConditionedPD())


def test_customer_raroc_equals_sum_then_divide_of_facilities():
    customer = make_customer()
    facility_a = wholesale_facility(customer=customer, facid="A")
    facility_b = wholesale_facility(customer=customer, facid="B", drawn_balance=500_000.0, limit=500_000.0)

    income_total = expected_loss_total = capital_total = 0.0
    as_of = START
    horizon_end = datetime(2021, 1, 1)
    for facility in (facility_a, facility_b):
        cashflows = project_facility_cashflows(facility, as_of, funding_model=FUNDING, opex_model=OPEX)
        income_total += sum(
            e.interest + e.commitment_fee + e.upfront_fee - e.funding_cost - e.operating_cost
            for e in cashflows.events
            if as_of <= e.date < horizon_end
        )
        expected_loss_total += ecl(facility, as_of)
        capital_total += regulatory_capital(facility, as_of)

    expected = (income_total - expected_loss_total) / capital_total
    result = customer_raroc(customer, FUNDING, OPEX, as_of_date=as_of)
    assert result == pytest.approx(expected)


def test_portfolio_raroc_equals_sum_then_divide_of_customers():
    from credit_portfolio.domain import Portfolio

    customer_1 = make_customer(customerid="C1")
    wholesale_facility(customer=customer_1)
    customer_2 = make_customer(customerid="C2")
    wholesale_facility(customer=customer_2, drawn_balance=300_000.0, limit=300_000.0)

    portfolio = Portfolio("P1", "Test portfolio")
    portfolio.add_customer(customer_1)
    portfolio.add_customer(customer_2)

    as_of = START
    # Compute the expected value via direct totals rather than averaging the
    # (non-linear) per-customer RAROC values, which can't simply be averaged.
    income_total = expected_loss_total = capital_total = 0.0
    horizon_end = datetime(2021, 1, 1)
    for customer in (customer_1, customer_2):
        for facility in customer.facility_list:
            cashflows = project_facility_cashflows(facility, as_of, funding_model=FUNDING, opex_model=OPEX)
            income_total += sum(
                e.interest + e.commitment_fee + e.upfront_fee - e.funding_cost - e.operating_cost
                for e in cashflows.events
                if as_of <= e.date < horizon_end
            )
            expected_loss_total += ecl(facility, as_of)
            capital_total += regulatory_capital(facility, as_of)
    expected = (income_total - expected_loss_total) / capital_total

    result = portfolio_raroc(portfolio, FUNDING, OPEX, as_of_date=as_of)
    assert result == pytest.approx(expected)


def test_horizon_changes_included_income():
    facility = wholesale_facility()
    one_year = facility_raroc(facility, FUNDING, OPEX, as_of_date=START, horizon=1)
    two_year = facility_raroc(facility, FUNDING, OPEX, as_of_date=START, horizon=2)
    assert one_year != pytest.approx(two_year)
