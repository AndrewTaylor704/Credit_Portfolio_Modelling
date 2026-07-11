"""Credit-conversion-factor / exposure-at-default models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from .scenario import Scenario

if TYPE_CHECKING:
    from credit_portfolio.domain.facility import Facility


class CCFModel(Protocol):
    def exposure(self, facility: "Facility", as_of_date: datetime | None, scenario: Scenario | None) -> float: ...


class StaticCCF:
    """Returns facility.limit unchanged — today's behaviour (implicitly
    CCF=1.0 applied to the full undrawn commitment)."""

    def exposure(self, facility: "Facility", as_of_date: datetime | None = None, scenario: Scenario | None = None) -> float:
        return facility.limit


_CCF_BY_TYPE = {
    "Loan": 1.0,
    "Term Loan": 1.0,
    "Revolver": 0.5,
    "RevolvingCredit": 0.5,
    "Overdraft": 0.2,
    "Guarantee": 1.0,
}
_DEFAULT_CCF = 1.0  # conservative fallback: matches StaticCCF's implicit CCF=1.0


class RegulatoryCCF:
    """exposure = drawn_balance + ccf * max(limit - drawn_balance, 0).

    ``ccf`` is looked up by facility.type from a small illustrative mapping,
    NOT a full Basel product-taxonomy CCF table — falls back to 1.0 (the
    conservative default, matching StaticCCF) for any unmapped type, so this
    is always a weakly-lower-or-equal exposure than today's behaviour, never
    a silently under-capitalised one.

    The undrawn amount is clamped to non-negative so an over-drawn facility
    (drawn_balance > limit) never produces an exposure below drawn_balance.
    """

    def exposure(self, facility: "Facility", as_of_date: datetime | None = None, scenario: Scenario | None = None) -> float:
        ccf = _CCF_BY_TYPE.get(facility.type, _DEFAULT_CCF)
        undrawn = max(facility.limit - facility.drawn_balance, 0.0)
        return facility.drawn_balance + ccf * undrawn
