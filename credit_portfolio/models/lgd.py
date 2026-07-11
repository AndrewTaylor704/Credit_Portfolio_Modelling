"""Loss-given-default models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from .scenario import Scenario

if TYPE_CHECKING:
    from credit_portfolio.domain.facility import Facility


class LGDModel(Protocol):
    def lgd(self, facility: "Facility", as_of_date: datetime | None, scenario: Scenario | None) -> float: ...


class StaticLGD:
    """Returns facility.lgd unchanged — today's behaviour, treated as the
    through-the-cycle (TTC) LGD."""

    def lgd(self, facility: "Facility", as_of_date: datetime | None = None, scenario: Scenario | None = None) -> float:
        return facility.lgd


class DownturnLGD:
    """downturn_lgd = min(facility.lgd + addon, 1.0).

    Deliberately NOT scenario-conditioned: Basel's downturn-LGD requirement
    is a fixed conservative floor/add-on applied once, not a live macro
    transform — the same regulatory design that keeps PD through-the-cycle
    in the IRB formula keeps LGD's systematic conditioning out of the
    per-scenario path entirely. Unlike ``MacroConditionedPD``, this is valid
    as an input to ecl() under any scenario, since ECL doesn't share the IRB
    formula's TTC-only constraint.
    """

    def __init__(self, addon: float):
        self.addon = addon

    def lgd(self, facility: "Facility", as_of_date: datetime | None = None, scenario: Scenario | None = None) -> float:
        return min(facility.lgd + self.addon, 1.0)
