from datetime import datetime

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

from credit_portfolio.amortisation import resolve as resolve_schedule
from credit_portfolio.domain import Portfolio
from credit_portfolio.risk import simulate_cashflow_paths
from tests.conftest import make_customer, make_facility

AS_OF = datetime(2020, 1, 1)


def test_result_shapes():
    portfolio = Portfolio("P1", "test")
    customer = make_customer(probdef=0.02)
    make_facility(customer=customer)
    portfolio.add_customer(customer)

    result = simulate_cashflow_paths(
        portfolio, correlation=0.15, num_sims=50, periods_per_year=4, horizon=2, as_of_date=AS_OF
    )
    assert result.scenario_cashflows.shape == (50, 8)
    assert result.scenario_losses.shape == (50, 8)
    assert result.scheduled_cashflow.shape == (8,)
    assert len(result.period_dates) == 8


def test_zero_pd_customers_always_receive_full_scheduled_cashflow():
    portfolio = Portfolio("P1", "test")
    customer = make_customer(probdef=0.0)
    make_facility(customer=customer)
    portfolio.add_customer(customer)

    result = simulate_cashflow_paths(
        portfolio, correlation=0.15, num_sims=20, periods_per_year=4, horizon=2, as_of_date=AS_OF
    )
    assert result.scenario_losses == pytest.approx(np.zeros_like(result.scenario_losses))
    for i in range(20):
        assert result.scenario_cashflows[i] == pytest.approx(result.scheduled_cashflow)


def test_certain_default_defaults_in_first_period_and_posts_recovery():
    portfolio = Portfolio("P1", "test")
    customer = make_customer(probdef=1.0)
    facility = make_facility(customer=customer, lgd=0.4)
    portfolio.add_customer(customer)

    recovery_lag = relativedelta(months=3)  # exactly one quarterly period
    result = simulate_cashflow_paths(
        portfolio, correlation=0.15, num_sims=5, periods_per_year=4, horizon=2,
        as_of_date=AS_OF, recovery_lag=recovery_lag,
    )

    resolved = resolve_schedule(facility.amort_schedule, facility.start_date, facility.maturity_date, facility.drawn_balance)
    balance_at_period_0 = resolved.balance_on_date(result.period_dates[0])
    expected_recovery = balance_at_period_0 * (1 - facility.lgd)
    expected_loss = balance_at_period_0 * facility.lgd

    for i in range(5):
        assert result.scenario_cashflows[i, 0] == pytest.approx(0.0)
        assert result.scenario_losses[i, 0] == pytest.approx(expected_loss)
        assert result.scenario_cashflows[i, 1] == pytest.approx(expected_recovery)
        assert result.scenario_cashflows[i, 2:] == pytest.approx(np.zeros(result.scenario_cashflows.shape[1] - 2))
        assert result.scenario_losses[i, 1:] == pytest.approx(np.zeros(result.scenario_losses.shape[1] - 1))
