import datetime as dt

import numpy as np
import pytest

from credit_portfolio.domain import Portfolio
from credit_portfolio.risk import simulate_loss_distribution
from credit_portfolio.valuation import customer_calc_losses
from tests.conftest import make_customer, make_facility

NUM_SIMS = 2000
NUM_BINS = 50
# Fixed, well within conftest's default facility life (start 2020-01-01,
# maturity 2027-01-01) so a 1-year horizon never lands past maturity
# regardless of wall-clock "now".
AS_OF_DATE = dt.datetime(2024, 1, 1)


def small_portfolio():
    portfolio = Portfolio("P1", "Small test portfolio")
    for i in range(5):
        customer = make_customer(customerid=f"C{i}", probdef=0.01 * (i + 1))
        make_facility(customer=customer, facid=f"F{i}", limit=1_000_000.0, drawn_balance=1_000_000.0)
        portfolio.add_customer(customer)
    return portfolio


def test_scenario_losses_has_one_entry_per_simulation():
    result = simulate_loss_distribution(
        small_portfolio(), correlation=0.15, num_sims=NUM_SIMS, num_bins=NUM_BINS, as_of_date=AS_OF_DATE
    )
    assert result.scenario_losses.shape == (NUM_SIMS,)


def test_total_notional_matches_sum_of_customer_losses_at_horizon():
    portfolio = small_portfolio()
    horizon_date = AS_OF_DATE + dt.timedelta(days=365.25)
    expected_total = sum(customer_calc_losses(c, horizon_date) for c in portfolio.customer_list)

    result = simulate_loss_distribution(
        portfolio, correlation=0.15, num_sims=NUM_SIMS, num_bins=NUM_BINS, as_of_date=AS_OF_DATE
    )
    assert result.total_notional == pytest.approx(expected_total)


def test_loss_dist_survival_counts_match_independent_recomputation_from_scenario_losses():
    result = simulate_loss_distribution(
        small_portfolio(), correlation=0.15, num_sims=NUM_SIMS, num_bins=NUM_BINS, as_of_date=AS_OF_DATE
    )

    bin_indices = np.minimum(
        (result.scenario_losses / result.total_notional * NUM_BINS).astype(int), NUM_BINS - 1
    )
    counts = np.bincount(bin_indices, minlength=NUM_BINS)
    expected_survival = np.cumsum(counts[::-1])[::-1] / NUM_SIMS

    assert result.loss_dist[:, 0] == pytest.approx(expected_survival)
