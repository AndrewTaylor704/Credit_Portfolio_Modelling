from datetime import datetime

import numpy as np
import pytest

from credit_portfolio.domain import Portfolio
from credit_portfolio.risk import simulate_cashflow_paths
from credit_portfolio.risk.cashflow_paths import CashflowPathsResult
from credit_portfolio.securitisation import Tranche, run_waterfall
from tests.conftest import make_customer, make_facility


def _stack():
    return [
        Tranche("equity", 0.0, 0.05),
        Tranche("mezzanine", 0.05, 0.15),
        Tranche("senior", 0.15, 1.0),
    ]


def test_no_default_every_tranche_keeps_full_notional_and_pro_rata_cash():
    num_sims, num_periods = 3, 4
    total_notional = 1_000_000.0
    scheduled = np.array([10_000.0, 10_000.0, 10_000.0, 10_000.0])
    result = CashflowPathsResult(
        period_dates=tuple(range(num_periods)),
        scheduled_cashflow=scheduled,
        scenario_cashflows=np.tile(scheduled, (num_sims, 1)),
        scenario_losses=np.zeros((num_sims, num_periods)),
        total_notional=total_notional,
    )
    tranches = _stack()
    out = run_waterfall(tranches, result)

    for tranche in tranches:
        tc = out[tranche.name]
        assert tc.remaining_notional == pytest.approx(np.full((num_sims, num_periods), tranche.thickness * total_notional))
        assert tc.cash == pytest.approx(tranche.thickness * result.scenario_cashflows)


def test_equity_wiped_out_receives_no_further_cash_while_senior_untouched():
    total_notional = 1_000_000.0
    tranches = _stack()  # equity 50k, mezzanine 100k, senior 850k notional

    # Period 0: 60k loss -> wipes equity's 50k, eats 10k into mezzanine.
    result = CashflowPathsResult(
        period_dates=(0, 1, 2),
        scheduled_cashflow=np.array([100_000.0, 100_000.0, 100_000.0]),
        scenario_cashflows=np.array([[100_000.0, 100_000.0, 100_000.0]]),
        scenario_losses=np.array([[60_000.0, 0.0, 0.0]]),
        total_notional=total_notional,
    )
    out = run_waterfall(tranches, result)

    assert out["equity"].remaining_notional[0] == pytest.approx([0.0, 0.0, 0.0])
    assert out["equity"].cash[0] == pytest.approx([0.0, 0.0, 0.0])
    assert out["mezzanine"].remaining_notional[0, 0] == pytest.approx(90_000.0)
    assert out["senior"].remaining_notional[0] == pytest.approx([850_000.0, 850_000.0, 850_000.0])
    assert out["senior"].cash[0] == pytest.approx(
        (850_000.0 / (850_000.0 + 90_000.0)) * np.array([100_000.0, 100_000.0, 100_000.0])
    )


def test_equity_cash_yield_is_lower_than_senior_end_to_end():
    portfolio = Portfolio("P1", "waterfall e2e")
    for i in range(8):
        customer = make_customer(customerid=f"C{i}", probdef=0.05 * (i + 1))
        make_facility(customer=customer, facid=f"F{i}", limit=1_000_000.0, drawn_balance=1_000_000.0)
        portfolio.add_customer(customer)

    result = simulate_cashflow_paths(
        portfolio, correlation=0.2, num_sims=300, periods_per_year=4, horizon=3, as_of_date=datetime(2020, 1, 1)
    )
    tranches = _stack()
    out = run_waterfall(tranches, result)

    def cash_yield(name: str) -> float:
        tranche = next(t for t in tranches if t.name == name)
        notional = tranche.thickness * result.total_notional
        return out[name].cash.sum() / notional

    assert cash_yield("equity") < cash_yield("senior")
