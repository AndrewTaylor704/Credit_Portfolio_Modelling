"""Operating-cost (cost-to-serve) models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from .scenario import Scenario

if TYPE_CHECKING:
    from credit_portfolio.domain.facility import Facility


class OperatingCostModel(Protocol):
    def cost_rate(self, facility: "Facility", as_of_date: datetime | None, scenario: Scenario | None) -> float: ...


class FlatOperatingCostRate:
    """A single annualized cost-to-serve rate on the opening balance.

    A simplification -- real cost-to-serve is often a fixed per-facility
    component plus a smaller balance-linked one -- kept as an extension
    point like ``FlatFundingRate``.
    """

    def __init__(self, rate: float):
        self.rate = rate

    def cost_rate(self, facility: "Facility", as_of_date: datetime | None = None, scenario: Scenario | None = None) -> float:
        return self.rate
