"""Tranche loss allocation for structured finance (cash securitisation /
synthetic risk transfer).

Deliberately simulation-agnostic: operates on a ``scenario_losses`` array
and a ``total_notional`` float, not on ``Portfolio``/``MonteCarloResult``
directly, so it composes with ``credit_portfolio.risk.montecarlo``'s output
without being coupled to it -- any per-scenario loss array works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Tranche:
    name: str
    attachment_point: float   # fraction of total notional, e.g. 0.0
    detachment_point: float    # fraction of total notional, e.g. 0.05 (a 5%-thick first-loss/equity tranche)

    def __post_init__(self) -> None:
        if not (0.0 <= self.attachment_point < self.detachment_point <= 1.0):
            raise ValueError(
                f"tranche {self.name!r} must satisfy 0 <= attachment_point < detachment_point <= 1, "
                f"got attachment_point={self.attachment_point}, detachment_point={self.detachment_point}"
            )

    @property
    def thickness(self) -> float:
        return self.detachment_point - self.attachment_point


def validate_tranche_stack(tranches: Sequence[Tranche]) -> None:
    """Raises ``ValueError`` unless the tranches' attachment/detachment
    points exactly partition [0, 1] with no gap or overlap."""
    if not tranches:
        raise ValueError("tranche stack is empty")

    ordered = sorted(tranches, key=lambda t: t.attachment_point)
    if ordered[0].attachment_point != 0.0:
        raise ValueError(f"tranche stack must start at 0.0, but the most junior tranche starts at {ordered[0].attachment_point}")
    if ordered[-1].detachment_point != 1.0:
        raise ValueError(f"tranche stack must end at 1.0, but the most senior tranche ends at {ordered[-1].detachment_point}")

    for lower, upper in zip(ordered, ordered[1:]):
        if lower.detachment_point < upper.attachment_point:
            raise ValueError(f"gap in tranche stack between {lower.name!r} (ends {lower.detachment_point}) and {upper.name!r} (starts {upper.attachment_point})")
        if lower.detachment_point > upper.attachment_point:
            raise ValueError(f"overlap in tranche stack between {lower.name!r} (ends {lower.detachment_point}) and {upper.name!r} (starts {upper.attachment_point})")


def allocate_losses(tranche: Tranche, scenario_losses: np.ndarray, total_notional: float) -> np.ndarray:
    """Per-scenario absolute loss amount absorbed by this tranche."""
    attachment_amount = tranche.attachment_point * total_notional
    tranche_notional = tranche.thickness * total_notional
    return np.clip(scenario_losses - attachment_amount, 0.0, tranche_notional)


def tranche_expected_loss_rate(tranche: Tranche, scenario_losses: np.ndarray, total_notional: float) -> float:
    """The tranche's expected loss as a fraction of its own notional (e.g.
    for sizing a synthetic-risk-transfer protection premium)."""
    tranche_notional = tranche.thickness * total_notional
    return float(np.mean(allocate_losses(tranche, scenario_losses, total_notional)) / tranche_notional)
