"""Single-factor Gaussian copula Monte Carlo economic capital / loss
distribution simulation, ported from the original ``Portfolio`` methods.

Single-period only (one horizon, binary default/no-default per customer).
Retains per-scenario portfolio losses (``MonteCarloResult.scenario_losses``)
alongside the binned survival histogram, which is what
``credit_portfolio.securitisation`` tranche *loss* allocation consumes. For
per-scenario, per-period *cashflows* (needed for the cash-securitisation
waterfall), see ``cashflow_paths.simulate_cashflow_paths``, which simulates
default *timing* across multiple periods rather than a single horizon.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from credit_portfolio.domain.currency import require_single_currency
from credit_portfolio.domain.portfolio import Portfolio
from credit_portfolio.valuation.aggregation import customer_calc_losses


@dataclass(frozen=True)
class MonteCarloResult:
    loss_dist: np.ndarray       # (num_bins, 2): column 0 is P(loss >= bin threshold), column 1 is the bin's loss threshold
    scenario_losses: np.ndarray  # (num_sims,): raw per-scenario portfolio loss amount
    total_notional: float          # sum of potential_loss across customers at the simulation horizon


def simulate_loss_distribution(
    portfolio: Portfolio,
    correlation: float,
    num_sims: int,
    num_bins: int,
    horizon: float = 1,
    random_seed: int = 1234,
    as_of_date: dt.datetime | None = None,
) -> MonteCarloResult:
    as_of_date = as_of_date or dt.datetime.now()
    horizon_date = as_of_date + dt.timedelta(days=horizon * 365.25)

    customers = portfolio.customer_list
    require_single_currency(f for c in customers for f in c.facility_list)
    num_customers = len(customers)
    threshold_value = np.array([norm.ppf(c.probdef) for c in customers])
    potential_loss = np.array([customer_calc_losses(c, horizon_date) for c in customers])
    total_notional = potential_loss.sum()

    rng = np.random.default_rng(seed=random_seed)
    scenario_losses = np.empty(num_sims)
    bin_indices = np.empty(num_sims, dtype=int)
    print_every = max(int(np.ceil(num_sims / 10)), 1)

    for i in range(num_sims):
        if (i + 1) % print_every == 0:
            print(f"Simulation # {i + 1}")
        syst_rand = rng.standard_normal()
        idio_rand = rng.standard_normal(num_customers)
        cust_asset_value = np.sqrt(correlation) * syst_rand + np.sqrt(1 - correlation) * idio_rand
        default_marker = cust_asset_value < threshold_value
        sim_loss = np.sum(default_marker * potential_loss)
        scenario_losses[i] = sim_loss
        bin_indices[i] = min(int(sim_loss / total_notional * num_bins), num_bins - 1)

    # loss_dist[j, 0] = fraction of simulations whose loss reached bin j or
    # beyond, i.e. a reverse-cumulative ("survival") count of the bin histogram.
    counts = np.bincount(bin_indices, minlength=num_bins)
    survival_counts = np.cumsum(counts[::-1])[::-1]

    binsize = total_notional / num_bins
    loss_dist = np.zeros((num_bins, 2))
    loss_dist[:, 0] = survival_counts / num_sims
    loss_dist[:, 1] = binsize * np.arange(num_bins)

    return MonteCarloResult(loss_dist=loss_dist, scenario_losses=scenario_losses, total_notional=total_notional)


def plot_loss_dist(loss_dist: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.plot(loss_dist[:, 1], loss_dist[:, 0])
    plt.xlabel("Loss amount")
    plt.ylabel("Probability")
    plt.title("MC Simulation of Portfolio Loss Distribution")
    plt.xlim(0, np.max(loss_dist[:, 1]) / 5)
    plt.show()


def loss_dist_quantile(loss_dist: np.ndarray, confidence: float):
    for i in range(loss_dist.shape[0] - 1):
        if loss_dist[i, 0] > 1 - confidence > loss_dist[i + 1, 0]:
            return loss_dist[i, 1]
    return False
