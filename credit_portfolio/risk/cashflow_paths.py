"""Multi-period single-factor Gaussian copula simulation of default TIMING
(not just whether a customer defaults within one horizon), producing
per-scenario, per-period portfolio cashflows and losses -- what a cash
securitisation waterfall needs.

Simplifications (see the design doc for the reasoning):
  - Flat-hazard default timing: each customer's annual PD is converted to a
    per-period marginal default probability via a constant-hazard
    assumption. Real term structures of default probability vary; this is
    a standard simplification for a model without a full PD curve.
  - Per-customer, per-period cashflow/loss/recovery amounts are precomputed
    ONCE outside the simulation loop (mirroring how the single-period
    engine in ``montecarlo.py`` precomputes ``potential_loss`` once) --
    the ``num_sims`` loop only does cheap vectorized lookups/sums over
    those precomputed tables, never re-resolving an amortisation schedule
    per scenario.
"""

from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass

import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

from credit_portfolio.amortisation import resolve as resolve_schedule
from credit_portfolio.cashflow import DEFAULT_RECOVERY_LAG, project_customer_cashflows
from credit_portfolio.domain.currency import require_single_currency
from credit_portfolio.domain.portfolio import Portfolio


@dataclass(frozen=True)
class CashflowPathsResult:
    period_dates: tuple[dt.datetime, ...]
    scheduled_cashflow: np.ndarray   # (num_periods,): portfolio cash if nobody defaults -- the tranches' "par" entitlement baseline
    scenario_cashflows: np.ndarray    # (num_sims, num_periods): actual aggregate cash collected per scenario per period
    scenario_losses: np.ndarray        # (num_sims, num_periods): principal loss recognised per scenario per period
    total_notional: float                 # total pool balance at as_of_date (NOT an LGD-weighted loss figure -- see montecarlo.MonteCarloResult for that)


def _period_dates(as_of_date: dt.datetime, num_periods: int, periods_per_year: int) -> list[dt.datetime]:
    months_per_period = round(12 / periods_per_year)
    dates = []
    current = as_of_date
    for _ in range(num_periods):
        current = current + relativedelta(months=months_per_period)
        dates.append(current)
    return dates


def _bucket_cashflow_by_period(schedule, as_of_date: dt.datetime, period_dates: list[dt.datetime]) -> np.ndarray:
    """Sums a cashflow schedule's non-recovery fields into period bins.
    Period t's window is (period_dates[t-1], period_dates[t]] (as_of_date
    stands in for period_dates[-1]); events at or before as_of_date (already
    paid, since ``project_customer_cashflows`` resolves a facility's full
    schedule from its own start_date, not from as_of_date) and events beyond
    the last period are dropped -- neither falls inside this simulated deal
    horizon."""
    num_periods = len(period_dates)
    row = np.zeros(num_periods)
    for event in schedule.events:
        if event.date <= as_of_date:
            continue
        idx = bisect.bisect_left(period_dates, event.date)
        if idx < num_periods:
            row[idx] += event.interest + event.commitment_fee + event.upfront_fee + event.principal
    return row


def simulate_cashflow_paths(
    portfolio: Portfolio,
    correlation: float,
    num_sims: int,
    periods_per_year: int,
    horizon: float,
    random_seed: int = 1234,
    as_of_date: dt.datetime | None = None,
    recovery_lag: relativedelta = DEFAULT_RECOVERY_LAG,
) -> CashflowPathsResult:
    as_of_date = as_of_date or dt.datetime.now()
    num_periods = round(horizon * periods_per_year)
    period_dates = _period_dates(as_of_date, num_periods, periods_per_year)
    period_length_years = 1.0 / periods_per_year

    customers = portfolio.customer_list
    require_single_currency(f for c in customers for f in c.facility_list)
    num_customers = len(customers)

    period_threshold = np.empty(num_customers)
    period_cash_matrix = np.zeros((num_customers, num_periods))
    loss_if_default_in_period = np.zeros((num_customers, num_periods))
    recovery_amount_by_default_period = np.zeros((num_customers, num_periods))
    total_notional = 0.0

    for ci, customer in enumerate(customers):
        period_pd = 1 - (1 - customer.probdef) ** period_length_years
        period_threshold[ci] = norm.ppf(period_pd)

        schedule = project_customer_cashflows(customer, as_of_date)
        period_cash_matrix[ci, :] = _bucket_cashflow_by_period(schedule, as_of_date, period_dates)

        for facility in customer.facility_list:
            resolved = resolve_schedule(facility.amort_schedule, facility.start_date, facility.maturity_date, facility.drawn_balance)
            total_notional += resolved.balance_on_date(as_of_date)
            for t in range(num_periods):
                balance_t = resolved.balance_on_date(period_dates[t])
                loss_if_default_in_period[ci, t] += balance_t * facility.lgd
                recovery_amount_by_default_period[ci, t] += balance_t * (1 - facility.lgd)

    recovery_target_index = np.array([
        bisect.bisect_left(period_dates, period_dates[t] + recovery_lag) for t in range(num_periods)
    ])
    scheduled_cashflow = period_cash_matrix.sum(axis=0)

    rng = np.random.default_rng(seed=random_seed)
    scenario_cashflows = np.zeros((num_sims, num_periods))
    scenario_losses = np.zeros((num_sims, num_periods))
    print_every = max(int(np.ceil(num_sims / 10)), 1)

    for i in range(num_sims):
        if (i + 1) % print_every == 0:
            print(f"Simulation # {i + 1}")
        alive = np.ones(num_customers, dtype=bool)
        cash_row = np.zeros(num_periods)
        loss_row = np.zeros(num_periods)

        for t in range(num_periods):
            syst_rand = rng.standard_normal()
            idio_rand = rng.standard_normal(num_customers)
            asset_value = np.sqrt(correlation) * syst_rand + np.sqrt(1 - correlation) * idio_rand
            newly_defaulted = alive & (asset_value < period_threshold)
            surviving = alive & ~newly_defaulted

            cash_row[t] += period_cash_matrix[surviving, t].sum()

            if newly_defaulted.any():
                loss_row[t] += loss_if_default_in_period[newly_defaulted, t].sum()
                target = recovery_target_index[t]
                if target < num_periods:
                    cash_row[target] += recovery_amount_by_default_period[newly_defaulted, t].sum()

            alive = surviving

        scenario_cashflows[i, :] = cash_row
        scenario_losses[i, :] = loss_row

    return CashflowPathsResult(
        period_dates=tuple(period_dates),
        scheduled_cashflow=scheduled_cashflow,
        scenario_cashflows=scenario_cashflows,
        scenario_losses=scenario_losses,
        total_notional=total_notional,
    )
