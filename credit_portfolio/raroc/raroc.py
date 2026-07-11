"""Risk-adjusted return on capital: a composition layer over the
cashflow, ECL, and capital modules -- no new simulation.

RAROC = (income - expected_loss) * (1 - tax_rate) / capital

``funding_model``/``opex_model`` are required (no default): a RAROC computed
with an implicit zero funding or operating cost would be misleading, so the
caller must decide rather than silently getting that answer.
``tax_rate`` defaults to 0.0 (pre-tax RAROC) -- a single scalar default of 0
is a well-understood explicit convention, unlike a missing cost model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dateutil.relativedelta import relativedelta

from credit_portfolio.cashflow.project import project_facility_cashflows
from credit_portfolio.domain.currency import require_single_currency
from credit_portfolio.domain.customer import Customer
from credit_portfolio.domain.facility import Facility
from credit_portfolio.domain.portfolio import Portfolio
from credit_portfolio.models import StaticCCF, StaticLGD, StaticPD
from credit_portfolio.valuation.ecl import ecl

from .capital import economic_capital, regulatory_capital

_DEFAULT_PD_MODEL = StaticPD()
_DEFAULT_LGD_MODEL = StaticLGD()
_DEFAULT_CCF_MODEL = StaticCCF()


def _facility_income_el_capital(
    facility: Facility,
    funding_model,
    opex_model,
    as_of_date: datetime,
    scenario: Any | None,
    horizon: float,
    capital_basis: str,
    pd_model,
    lgd_model,
    ccf_model,
    target_capital_ratio: float,
    confidence: float,
) -> tuple[float, float, float]:
    horizon_end = as_of_date + relativedelta(years=horizon)
    cashflows = project_facility_cashflows(
        facility, as_of_date, scenario, lgd_model, funding_model=funding_model, opex_model=opex_model
    )
    income = sum(
        e.interest + e.commitment_fee + e.upfront_fee - e.funding_cost - e.operating_cost
        for e in cashflows.events
        if as_of_date <= e.date < horizon_end
    )
    expected_loss = ecl(facility, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model, ccf_model=ccf_model)
    if capital_basis == "regulatory":
        capital = regulatory_capital(facility, as_of_date, scenario, pd_model, lgd_model, ccf_model, target_capital_ratio)
    elif capital_basis == "economic":
        capital = economic_capital(facility, as_of_date, scenario, pd_model, lgd_model, ccf_model, confidence)
    else:
        raise ValueError(f"capital_basis must be 'regulatory' or 'economic', got {capital_basis!r}")
    return income, expected_loss, capital


def _raroc_from_totals(income: float, expected_loss: float, capital: float, tax_rate: float) -> float:
    pretax_numerator = income - expected_loss
    posttax_numerator = pretax_numerator * (1 - tax_rate)
    return posttax_numerator / capital


def facility_raroc(
    facility: Facility,
    funding_model,
    opex_model,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    horizon: float = 1,
    capital_basis: str = "regulatory",
    tax_rate: float = 0.0,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
    target_capital_ratio: float = 0.08,
    confidence: float = 0.999,
) -> float:
    as_of_date = as_of_date or datetime.now()
    income, expected_loss, capital = _facility_income_el_capital(
        facility, funding_model, opex_model, as_of_date, scenario, horizon,
        capital_basis, pd_model, lgd_model, ccf_model, target_capital_ratio, confidence,
    )
    return _raroc_from_totals(income, expected_loss, capital, tax_rate)


def customer_raroc(
    customer: Customer,
    funding_model,
    opex_model,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    horizon: float = 1,
    capital_basis: str = "regulatory",
    tax_rate: float = 0.0,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
    target_capital_ratio: float = 0.08,
    confidence: float = 0.999,
) -> float:
    """Sums income, expected loss, and capital across the customer's
    facilities before dividing -- no diversification credit on the capital
    sum (sum-of-standalone, not portfolio-diversified; see economic_capital)."""
    require_single_currency(customer.facility_list)
    as_of_date = as_of_date or datetime.now()
    totals = [0.0, 0.0, 0.0]
    for facility in customer.facility_list:
        income, expected_loss, capital = _facility_income_el_capital(
            facility, funding_model, opex_model, as_of_date, scenario, horizon,
            capital_basis, pd_model, lgd_model, ccf_model, target_capital_ratio, confidence,
        )
        totals[0] += income
        totals[1] += expected_loss
        totals[2] += capital
    return _raroc_from_totals(*totals, tax_rate)


def portfolio_raroc(
    portfolio: Portfolio,
    funding_model,
    opex_model,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    horizon: float = 1,
    capital_basis: str = "regulatory",
    tax_rate: float = 0.0,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
    target_capital_ratio: float = 0.08,
    confidence: float = 0.999,
) -> float:
    """Sums income, expected loss, and capital across every customer's
    facilities before dividing -- no diversification credit, same
    simplification as customer_raroc/economic_capital."""
    require_single_currency(f for c in portfolio.customer_list for f in c.facility_list)
    as_of_date = as_of_date or datetime.now()
    totals = [0.0, 0.0, 0.0]
    for customer in portfolio.customer_list:
        for facility in customer.facility_list:
            income, expected_loss, capital = _facility_income_el_capital(
                facility, funding_model, opex_model, as_of_date, scenario, horizon,
                capital_basis, pd_model, lgd_model, ccf_model, target_capital_ratio, confidence,
            )
            totals[0] += income
            totals[1] += expected_loss
            totals[2] += capital
    return _raroc_from_totals(*totals, tax_rate)
