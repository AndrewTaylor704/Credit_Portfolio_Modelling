"""Funding-cost models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from .scenario import Scenario

if TYPE_CHECKING:
    from credit_portfolio.domain.facility import Facility


class FundingModel(Protocol):
    def funding_rate(self, facility: "Facility", as_of_date: datetime | None, scenario: Scenario | None) -> float: ...


class FlatFundingRate:
    """A single annualized rate applied uniformly regardless of tenor/date.

    A simplification -- a real funding curve would vary by tenor -- kept as
    an extension point (same pattern as ``MacroConditionedPD``'s Z-score:
    the interface exists now, a richer calibrated model can replace it
    later without changing call sites).
    """

    def __init__(self, rate: float):
        self.rate = rate

    def funding_rate(self, facility: "Facility", as_of_date: datetime | None = None, scenario: Scenario | None = None) -> float:
        return self.rate
