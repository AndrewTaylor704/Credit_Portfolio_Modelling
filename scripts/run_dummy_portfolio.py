"""Demo driver: loads the dummy loan-level CSV, prints portfolio RWA/UL,
runs the Monte Carlo loss simulation, and plots the loss distribution.

This is the moved-out, __main__-guarded equivalent of the old bottom-of-file
script in CreditObjects.py — importing credit_portfolio no longer runs a
simulation as a side effect.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from credit_portfolio.io import load_portfolio
from credit_portfolio.risk import loss_dist_quantile, plot_loss_dist, simulate_loss_distribution
from credit_portfolio.valuation import portfolio_rwa

DATA_PATH = REPO_ROOT / "Dummy_loan_data.csv"


def main() -> None:
    portfolio = load_portfolio(str(DATA_PATH), portfolio_id="10001", portfolio_name="Dummy portfolio")

    rwa = portfolio_rwa(portfolio)
    print(f"rwa = {rwa:.2f}")
    print(f"UL = {rwa / 12.5:.2f}")

    result = simulate_loss_distribution(portfolio, correlation=0.15, num_sims=100_000, num_bins=10_000)
    print(loss_dist_quantile(result.loss_dist, 0.999))
    plot_loss_dist(result.loss_dist)


if __name__ == "__main__":
    main()
