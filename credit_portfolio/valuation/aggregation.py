"""Fold facility-level valuation up to customer and portfolio level, and
resolve a facility's amortisation schedule for balance/loss queries as of
an explicit date. Nothing here caches anything — every call recomputes
from the as-of date given, which is what makes revaluation under a
different date or scenario possible.

Every aggregation below calls ``require_single_currency`` first -- summing
across facilities in different currencies without FX conversion (which
this codebase does not implement) would silently produce a meaningless
number.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from credit_portfolio.amortisation import resolve as resolve_schedule
from credit_portfolio.domain.currency import require_single_currency
from credit_portfolio.domain.customer import Customer
from credit_portfolio.domain.facility import Facility
from credit_portfolio.domain.portfolio import Portfolio
from credit_portfolio.models import StaticCCF, StaticLGD, StaticPD

from .ead import ead
from .ecl import ecl, ifrs9_ecl
from .rwa import rwa

_DEFAULT_PD_MODEL = StaticPD()
_DEFAULT_LGD_MODEL = StaticLGD()
_DEFAULT_CCF_MODEL = StaticCCF()


def _all_facilities(portfolio: Portfolio):
    return (f for c in portfolio.customer_list for f in c.facility_list)


def facility_balance_on_date(facility: Facility, as_of_date: datetime) -> float:
    schedule = resolve_schedule(facility.amort_schedule, facility.start_date, facility.maturity_date, facility.drawn_balance)
    return schedule.balance_on_date(as_of_date)


def facility_calc_losses(facility: Facility, as_of_date: datetime) -> float:
    return facility_balance_on_date(facility, as_of_date) * facility.lgd


def customer_ead(
    customer: Customer, as_of_date: datetime | None = None, scenario: Any | None = None, ccf_model=_DEFAULT_CCF_MODEL
) -> float:
    require_single_currency(customer.facility_list)
    return sum(ead(f, as_of_date, scenario, ccf_model=ccf_model) for f in customer.facility_list)


def customer_ecl(
    customer: Customer,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    require_single_currency(customer.facility_list)
    return sum(
        ecl(f, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model, ccf_model=ccf_model)
        for f in customer.facility_list
    )


def customer_ifrs9_ecl(
    customer: Customer,
    as_of_date: datetime,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    require_single_currency(customer.facility_list)
    return sum(
        ifrs9_ecl(f, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model, ccf_model=ccf_model)
        for f in customer.facility_list
    )


def customer_rwa(
    customer: Customer,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    require_single_currency(customer.facility_list)
    return sum(
        rwa(f, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model, ccf_model=ccf_model)
        for f in customer.facility_list
    )


def customer_risk_weight(
    customer: Customer,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    return customer_rwa(customer, as_of_date, scenario, pd_model, lgd_model, ccf_model) / customer_ead(
        customer, as_of_date, scenario, ccf_model
    )


def customer_balance_on_date(customer: Customer, as_of_date: datetime) -> float:
    require_single_currency(customer.facility_list)
    return sum(facility_balance_on_date(f, as_of_date) for f in customer.facility_list)


def customer_calc_losses(customer: Customer, as_of_date: datetime) -> float:
    require_single_currency(customer.facility_list)
    return sum(facility_calc_losses(f, as_of_date) for f in customer.facility_list)


def portfolio_ead(
    portfolio: Portfolio, as_of_date: datetime | None = None, scenario: Any | None = None, ccf_model=_DEFAULT_CCF_MODEL
) -> float:
    require_single_currency(_all_facilities(portfolio))
    return sum(customer_ead(c, as_of_date, scenario, ccf_model=ccf_model) for c in portfolio.customer_list)


def portfolio_ecl(
    portfolio: Portfolio,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    require_single_currency(_all_facilities(portfolio))
    return sum(
        customer_ecl(c, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model, ccf_model=ccf_model)
        for c in portfolio.customer_list
    )


def portfolio_ifrs9_ecl(
    portfolio: Portfolio,
    as_of_date: datetime,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    require_single_currency(_all_facilities(portfolio))
    return sum(
        customer_ifrs9_ecl(c, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model, ccf_model=ccf_model)
        for c in portfolio.customer_list
    )


def portfolio_rwa(
    portfolio: Portfolio,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    require_single_currency(_all_facilities(portfolio))
    return sum(
        customer_rwa(c, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model, ccf_model=ccf_model)
        for c in portfolio.customer_list
    )


def portfolio_risk_weight(
    portfolio: Portfolio,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    return portfolio_rwa(portfolio, as_of_date, scenario, pd_model, lgd_model, ccf_model) / portfolio_ead(
        portfolio, as_of_date, scenario, ccf_model
    )


def portfolio_balance_on_date(portfolio: Portfolio, as_of_date: datetime) -> float:
    require_single_currency(_all_facilities(portfolio))
    return sum(customer_balance_on_date(c, as_of_date) for c in portfolio.customer_list)


def portfolio_calc_losses(portfolio: Portfolio, as_of_date: datetime) -> float:
    require_single_currency(_all_facilities(portfolio))
    return sum(customer_calc_losses(c, as_of_date) for c in portfolio.customer_list)
