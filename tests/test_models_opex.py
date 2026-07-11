from datetime import datetime

import pytest

from credit_portfolio.amortisation import AmortisationSchedule, Frequency
from credit_portfolio.cashflow import project_facility_cashflows
from credit_portfolio.models import FlatOperatingCostRate
from tests.conftest import make_facility

START = datetime(2020, 1, 1)
MATURITY = datetime(2022, 1, 1)


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


def test_flat_operating_cost_rate_produces_expected_first_period_cost():
    facility = revolver_facility()
    schedule = project_facility_cashflows(facility, opex_model=FlatOperatingCostRate(0.005))

    first = schedule.events[0]
    period_fraction = (first.date - START).days / 365.0
    assert first.operating_cost == pytest.approx(0.005 * 600_000.0 * period_fraction)


def test_no_opex_model_reproduces_zero_operating_cost():
    facility = revolver_facility()
    schedule = project_facility_cashflows(facility)
    assert all(e.operating_cost == 0.0 for e in schedule.events)
