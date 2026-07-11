"""Structured finance (cash securitisation / synthetic risk transfer).

Tranche loss allocation (``tranche.py``) supports synthetic risk transfer
(sizing protection premiums/payouts from ``tranche_expected_loss_rate``).

The cash securitisation cashflow waterfall (``waterfall.py``) is implemented
as a pro-rata pass-through with reverse-seniority loss absorption, driven by
``risk.cashflow_paths.simulate_cashflow_paths``'s multi-period per-scenario
cashflows/losses.

NOT IMPLEMENTED: a full sequential-pay waterfall with independent
per-tranche coupons and principal-paydown priority (common in CLOs), and
trigger-based switching between pro-rata and sequential structures (a
future extension the per-period distribution rule in ``waterfall.py`` is
kept separable to support).
"""

from .tranche import Tranche, allocate_losses, tranche_expected_loss_rate, validate_tranche_stack
from .waterfall import TrancheCashflows, run_waterfall

__all__ = [
    "Tranche",
    "validate_tranche_stack",
    "allocate_losses",
    "tranche_expected_loss_rate",
    "run_waterfall",
    "TrancheCashflows",
]
