import datetime as dt

import numpy as np
import pytest

from credit_portfolio.domain import Portfolio
from credit_portfolio.risk import simulate_loss_distribution
from credit_portfolio.securitisation import Tranche, allocate_losses, tranche_expected_loss_rate, validate_tranche_stack
from tests.conftest import make_customer, make_facility


def test_tranche_rejects_attachment_at_or_above_detachment():
    with pytest.raises(ValueError):
        Tranche(name="bad", attachment_point=0.1, detachment_point=0.1)
    with pytest.raises(ValueError):
        Tranche(name="bad", attachment_point=0.2, detachment_point=0.1)


def test_tranche_rejects_out_of_unit_range_points():
    with pytest.raises(ValueError):
        Tranche(name="bad", attachment_point=-0.1, detachment_point=0.5)
    with pytest.raises(ValueError):
        Tranche(name="bad", attachment_point=0.5, detachment_point=1.1)


def _stack():
    return [
        Tranche("equity", 0.0, 0.05),
        Tranche("mezzanine", 0.05, 0.15),
        Tranche("senior", 0.15, 1.0),
    ]


def test_validate_tranche_stack_accepts_a_clean_partition():
    validate_tranche_stack(_stack())  # should not raise


def test_validate_tranche_stack_rejects_a_gap():
    stack = [Tranche("equity", 0.0, 0.05), Tranche("senior", 0.10, 1.0)]
    with pytest.raises(ValueError, match="gap"):
        validate_tranche_stack(stack)


def test_validate_tranche_stack_rejects_an_overlap():
    stack = [Tranche("equity", 0.0, 0.10), Tranche("senior", 0.05, 1.0)]
    with pytest.raises(ValueError, match="overlap"):
        validate_tranche_stack(stack)


def test_allocate_losses_clip_formula():
    total_notional = 1_000_000.0
    tranche = Tranche("mezzanine", attachment_point=0.05, detachment_point=0.15)  # 50k -> 150k, thickness 100k
    scenario_losses = np.array([0.0, 30_000.0, 100_000.0, 500_000.0])

    result = allocate_losses(tranche, scenario_losses, total_notional)
    # below attachment -> 0; inside band -> partial; at/above detachment -> full tranche wipeout (100k)
    assert result == pytest.approx([0.0, 0.0, 50_000.0, 100_000.0])


def test_tranche_expected_loss_rate_matches_hand_computation():
    total_notional = 1_000_000.0
    tranche = Tranche("mezzanine", attachment_point=0.05, detachment_point=0.15)
    scenario_losses = np.array([0.0, 30_000.0, 100_000.0, 500_000.0])

    expected = np.mean([0.0, 0.0, 50_000.0, 100_000.0]) / 100_000.0
    assert tranche_expected_loss_rate(tranche, scenario_losses, total_notional) == pytest.approx(expected)


def test_expected_loss_rate_decreases_by_seniority_end_to_end():
    portfolio = Portfolio("P1", "Tranching test portfolio")
    for i in range(8):
        customer = make_customer(customerid=f"C{i}", probdef=0.02 * (i + 1))
        make_facility(customer=customer, facid=f"F{i}", limit=1_000_000.0, drawn_balance=1_000_000.0)
        portfolio.add_customer(customer)

    result = simulate_loss_distribution(
        portfolio, correlation=0.2, num_sims=3000, num_bins=50, as_of_date=dt.datetime(2024, 1, 1)
    )

    equity, mezzanine, senior = _stack()
    equity_el = tranche_expected_loss_rate(equity, result.scenario_losses, result.total_notional)
    mezzanine_el = tranche_expected_loss_rate(mezzanine, result.scenario_losses, result.total_notional)
    senior_el = tranche_expected_loss_rate(senior, result.scenario_losses, result.total_notional)

    assert equity_el > mezzanine_el > senior_el
