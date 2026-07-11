from datetime import datetime

import pytest
from dateutil.relativedelta import relativedelta

from credit_portfolio.amortisation import AmortisationSchedule, Frequency
from credit_portfolio.cashflow import (
    CashflowEvent,
    CashflowSchedule,
    merge_cashflow_schedules,
    portfolio_cashflows,
    project_customer_cashflows,
    project_facility_cashflows,
)
from tests.conftest import make_customer, make_facility

START = datetime(2020, 1, 1)
MATURITY = datetime(2022, 1, 1)  # 2 years, 8 quarters


def revolver_facility(**overrides):
    defaults = dict(
        type="Revolver",
        start_date=START,
        maturity_date=MATURITY,
        limit=1_000_000.0,
        drawn_balance=600_000.0,
        margin=0.03,
        fee=0.01,
        amort_schedule=AmortisationSchedule.straight_line(Frequency.QUARTERLY),
    )
    defaults.update(overrides)
    return make_facility(**defaults)


def test_no_default_first_period_interest_and_commitment_fee():
    facility = revolver_facility()
    schedule = project_facility_cashflows(facility)

    first = schedule.events[0]
    period_fraction = (first.date - START).days / 365.0
    assert first.interest == pytest.approx(facility.margin * 600_000.0 * period_fraction)
    assert first.commitment_fee == pytest.approx(facility.fee * (1_000_000.0 - 600_000.0) * period_fraction)


def test_no_default_principal_sums_to_drawn_balance():
    facility = revolver_facility()
    schedule = project_facility_cashflows(facility)
    assert sum(e.principal for e in schedule.events) == pytest.approx(facility.drawn_balance)


def test_upfront_fee_accretes_to_the_full_amount_over_the_facility_life():
    facility = revolver_facility(upfront_fee_rate=0.02)
    schedule = project_facility_cashflows(facility)
    assert sum(e.upfront_fee for e in schedule.events) == pytest.approx(0.02 * facility.limit)


def test_no_upfront_fee_rate_means_no_upfront_fee_income():
    facility = revolver_facility()
    schedule = project_facility_cashflows(facility)
    assert sum(e.upfront_fee for e in schedule.events) == 0.0


def test_default_path_truncates_and_posts_a_single_recovery_event():
    facility = revolver_facility(lgd=0.4)
    default_date = datetime(2021, 1, 1)  # one year in
    recovery_lag = relativedelta(months=6)

    no_default = project_facility_cashflows(facility)
    defaulted = project_facility_cashflows(facility, default_date=default_date, recovery_lag=recovery_lag)

    kept_dates = [e.date for e in defaulted.events[:-1]]
    assert all(d < default_date for d in kept_dates)
    assert kept_dates == [e.date for e in no_default.events if e.date < default_date]

    recovery_event = defaulted.events[-1]
    assert recovery_event.date == default_date + recovery_lag
    assert recovery_event.principal == 0.0
    assert recovery_event.interest == 0.0

    from credit_portfolio.amortisation import resolve as resolve_schedule

    amort = resolve_schedule(facility.amort_schedule, facility.start_date, facility.maturity_date, facility.drawn_balance)
    balance_at_default = amort.balance_on_date(default_date)
    assert recovery_event.recovery == pytest.approx(balance_at_default * (1 - facility.lgd))


def test_default_on_or_after_maturity_leaves_schedule_unchanged():
    facility = revolver_facility()
    no_default = project_facility_cashflows(facility)
    after_maturity = project_facility_cashflows(facility, default_date=MATURITY)
    assert after_maturity == no_default

    after_maturity_2 = project_facility_cashflows(facility, default_date=MATURITY + relativedelta(years=1))
    assert after_maturity_2 == no_default


def test_customer_level_default_applies_to_every_facility_cross_default():
    customer = make_customer()
    facility_a = revolver_facility(customer=customer, facid="A")
    facility_b = revolver_facility(customer=customer, facid="B", drawn_balance=200_000.0)
    default_date = datetime(2021, 1, 1)

    result = project_customer_cashflows(customer, default_date=default_date)

    # Both facilities' recoveries should appear (merged onto the same date if
    # using the default recovery lag), proving facility B was also treated as
    # defaulted purely because facility A's default_date was applied to it.
    total_recovery = sum(e.recovery for e in result.events)
    assert total_recovery > 0
    assert all(e.date < default_date or e.recovery > 0 for e in result.events)


def test_merge_cashflow_schedules_sums_overlapping_and_non_overlapping_dates():
    d1, d2, d3 = datetime(2020, 1, 1), datetime(2020, 4, 1), datetime(2020, 7, 1)
    schedule_a = CashflowSchedule(events=(
        CashflowEvent(date=d1, interest=10.0),
        CashflowEvent(date=d2, interest=20.0),
    ))
    schedule_b = CashflowSchedule(events=(
        CashflowEvent(date=d2, interest=5.0),
        CashflowEvent(date=d3, interest=7.0),
    ))
    merged = merge_cashflow_schedules([schedule_a, schedule_b])

    by_date = {e.date: e.interest for e in merged.events}
    assert by_date == {d1: 10.0, d2: 25.0, d3: 7.0}


def test_portfolio_cashflows_no_default_equals_sum_of_customer_schedules():
    customer_1 = make_customer(customerid="C1")
    revolver_facility(customer=customer_1)
    customer_2 = make_customer(customerid="C2")
    revolver_facility(customer=customer_2, drawn_balance=300_000.0)

    from credit_portfolio.domain import Portfolio

    portfolio = Portfolio("P1", "Test portfolio")
    portfolio.add_customer(customer_1)
    portfolio.add_customer(customer_2)

    expected = merge_cashflow_schedules([
        project_customer_cashflows(customer_1),
        project_customer_cashflows(customer_2),
    ])
    assert portfolio_cashflows(portfolio) == expected
