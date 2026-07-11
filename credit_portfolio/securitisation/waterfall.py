"""Cash securitisation waterfall: pro-rata pass-through with reverse-
seniority loss absorption, driven by ``risk.cashflow_paths.CashflowPathsResult``.

NOT a full sequential-pay CDO waterfall with independent per-tranche
coupons and principal-paydown priority (common in CLOs) -- that needs each
tranche to carry its own coupon rate and an interest-then-principal-by-
seniority cascade each period, a further, separable piece of work. Pro-rata
pass-through (each tranche receives a share of actual portfolio cash
proportional to its own remaining, loss-eroded notional) is a real, common
structure (typical of many RMBS/ABS deals), just a simpler one.

The per-period distribution rule (``_pro_rata_shares``) is kept as its own
function rather than inlined into ``run_waterfall``'s loop: some real deals
switch from pro-rata to sequential amortisation once a performance trigger
breaches a threshold (and some can revert if performance cures). That
trigger-aware switching isn't built yet, but keeping the distribution rule
at one seam means a future version can swap it in -- or choose between
rules per period -- while reusing the same cumulative-loss/remaining-
notional bookkeeping below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from credit_portfolio.risk.cashflow_paths import CashflowPathsResult

from .tranche import Tranche, allocate_losses, validate_tranche_stack


@dataclass(frozen=True)
class TrancheCashflows:
    cash: np.ndarray                # (num_sims, num_periods): cash received by this tranche
    remaining_notional: np.ndarray    # (num_sims, num_periods): this tranche's notional after that period's losses


def _pro_rata_shares(remaining_notionals: np.ndarray) -> np.ndarray:
    """remaining_notionals: (num_tranches, num_sims) -> shares of the same
    shape, summing to 1 down each scenario column (or all zero for a
    scenario where every tranche has been wiped out)."""
    totals = remaining_notionals.sum(axis=0, keepdims=True)
    return np.divide(
        remaining_notionals, totals, out=np.zeros_like(remaining_notionals), where=totals > 0
    )


def run_waterfall(tranches: Sequence[Tranche], result: CashflowPathsResult) -> dict[str, TrancheCashflows]:
    validate_tranche_stack(tranches)

    num_sims, num_periods = result.scenario_cashflows.shape
    tranche_notionals = [t.thickness * result.total_notional for t in tranches]

    cash = {t.name: np.zeros((num_sims, num_periods)) for t in tranches}
    remaining_notional = {t.name: np.zeros((num_sims, num_periods)) for t in tranches}

    cumulative_loss = np.zeros(num_sims)
    for period_t in range(num_periods):
        cumulative_loss = cumulative_loss + result.scenario_losses[:, period_t]

        remaining_matrix = np.stack([
            notional - allocate_losses(tranche, cumulative_loss, result.total_notional)
            for tranche, notional in zip(tranches, tranche_notionals)
        ])  # (num_tranches, num_sims)

        for tranche, row in zip(tranches, remaining_matrix):
            remaining_notional[tranche.name][:, period_t] = row

        shares = _pro_rata_shares(remaining_matrix)  # (num_tranches, num_sims)
        for tranche, share_row in zip(tranches, shares):
            cash[tranche.name][:, period_t] = share_row * result.scenario_cashflows[:, period_t]

    return {
        tranche.name: TrancheCashflows(cash=cash[tranche.name], remaining_notional=remaining_notional[tranche.name])
        for tranche in tranches
    }
