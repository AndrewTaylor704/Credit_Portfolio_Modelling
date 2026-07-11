"""Projects income + principal cashflows for a facility (or a customer's
whole book, applying one shared default date across all their facilities --
the cross-default / obligor-level default convention)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from dateutil.relativedelta import relativedelta

from credit_portfolio.amortisation import resolve as resolve_schedule
from credit_portfolio.domain.currency import require_single_currency
from credit_portfolio.domain.customer import Customer
from credit_portfolio.domain.facility import Facility
from credit_portfolio.models import StaticLGD

from .schedule import CashflowEvent, CashflowSchedule, merge_cashflow_schedules

_DEFAULT_LGD_MODEL = StaticLGD()
DEFAULT_RECOVERY_LAG = relativedelta(years=1)


def project_facility_cashflows(
    facility: Facility,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    lgd_model=_DEFAULT_LGD_MODEL,
    default_date: date | None = None,
    recovery_lag: relativedelta = DEFAULT_RECOVERY_LAG,
    funding_model=None,
    opex_model=None,
) -> CashflowSchedule:
    amort = resolve_schedule(facility.amort_schedule, facility.start_date, facility.maturity_date, facility.drawn_balance)
    total_life_years = (facility.maturity_date - facility.start_date).days / 365.0
    total_upfront_fee = (facility.upfront_fee_rate or 0.0) * facility.limit

    events = []
    previous_date = facility.start_date
    for payment in amort.payments:
        period_fraction = (payment.date - previous_date).days / 365.0
        previous_date = payment.date

        interest = facility.margin * payment.opening_balance * period_fraction
        commitment_fee = facility.fee * (facility.limit - payment.opening_balance) * period_fraction
        upfront_fee = total_upfront_fee * (period_fraction / total_life_years) if total_upfront_fee else 0.0
        funding_cost = (
            funding_model.funding_rate(facility, payment.date, scenario) * payment.opening_balance * period_fraction
            if funding_model is not None
            else 0.0
        )
        operating_cost = (
            opex_model.cost_rate(facility, payment.date, scenario) * payment.opening_balance * period_fraction
            if opex_model is not None
            else 0.0
        )

        events.append(CashflowEvent(
            date=payment.date,
            interest=interest,
            commitment_fee=commitment_fee,
            upfront_fee=upfront_fee,
            principal=payment.principal_payment,
            funding_cost=funding_cost,
            operating_cost=operating_cost,
        ))

    if default_date is None or default_date >= facility.maturity_date:
        return CashflowSchedule(events=tuple(events))

    kept = [e for e in events if e.date < default_date]
    lgd = lgd_model.lgd(facility, default_date, scenario)
    balance_at_default = amort.balance_on_date(default_date)
    recovery_event = CashflowEvent(
        date=default_date + recovery_lag,
        recovery=balance_at_default * (1 - lgd),
    )
    kept.append(recovery_event)
    return CashflowSchedule(events=tuple(kept))


def project_customer_cashflows(
    customer: Customer,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    lgd_model=_DEFAULT_LGD_MODEL,
    default_date: date | None = None,
    recovery_lag: relativedelta = DEFAULT_RECOVERY_LAG,
    funding_model=None,
    opex_model=None,
) -> CashflowSchedule:
    """Applies the same default_date to every facility this customer holds
    (cross-default / obligor-level default convention: one facility
    defaulting means the customer has defaulted, not just that facility)."""
    require_single_currency(customer.facility_list)
    per_facility = [
        project_facility_cashflows(
            f, as_of_date, scenario, lgd_model, default_date, recovery_lag,
            funding_model=funding_model, opex_model=opex_model,
        )
        for f in customer.facility_list
    ]
    return merge_cashflow_schedules(per_facility)
