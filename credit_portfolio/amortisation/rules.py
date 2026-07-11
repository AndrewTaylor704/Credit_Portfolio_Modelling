"""Declarative building blocks for facility amortisation schedules.

A schedule is expressed as a template (see ``schedule.py``) built from
``AmortRule`` segments. Each rule's window is anchored relative to the
facility's start or maturity date (``RelativeAnchor``) rather than to
absolute dates, so the same rule can be resolved against any facility's
actual start/maturity once known.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import Literal

from dateutil.relativedelta import relativedelta


class AmortMethod(Enum):
    """How principal is paid down within a rule's window."""

    BULLET = auto()
    STRAIGHT_LINE = auto()
    PERCENT_OF_ORIGINAL = auto()
    PERCENT_OF_CURRENT = auto()
    FIXED_AMOUNT = auto()


class Frequency(Enum):
    """Payment cadence within a rule's window. Unused for BULLET."""

    MONTHLY = auto()
    QUARTERLY = auto()
    ANNUAL = auto()

    @property
    def step(self) -> relativedelta:
        return {
            Frequency.MONTHLY: relativedelta(months=1),
            Frequency.QUARTERLY: relativedelta(months=3),
            Frequency.ANNUAL: relativedelta(years=1),
        }[self]


@dataclass(frozen=True)
class RelativeAnchor:
    """A date expressed relative to a facility's start or maturity date.

    ``offset`` is added to the reference date, so ``from_maturity(years=-2)``
    means "two years before maturity" and ``from_start()`` means "at start".
    """

    reference: Literal["start", "maturity"]
    offset: relativedelta = relativedelta()

    @staticmethod
    def from_start(offset: relativedelta | None = None, **kwargs) -> "RelativeAnchor":
        return RelativeAnchor(reference="start", offset=offset if offset is not None else relativedelta(**kwargs))

    @staticmethod
    def from_maturity(offset: relativedelta | None = None, **kwargs) -> "RelativeAnchor":
        return RelativeAnchor(reference="maturity", offset=offset if offset is not None else relativedelta(**kwargs))

    def resolve(self, start_date: date, maturity_date: date) -> date:
        base = start_date if self.reference == "start" else maturity_date
        return base + self.offset


@dataclass(frozen=True)
class AmortRule:
    """One segment of an amortisation schedule.

    The segment's active window is ``(window_start, window_end]`` — a
    payment posts at each sub-period's end, never at the window's own start.
    This is what makes adjacent rules tile cleanly regardless of frequency
    mismatches at the seam.
    """

    method: AmortMethod
    window_start: RelativeAnchor | None = None
    window_end: RelativeAnchor | None = None
    frequency: Frequency | None = None
    rate: float | None = None
    amount: float | None = None

    def __post_init__(self) -> None:
        if self.method is not AmortMethod.BULLET and self.frequency is None:
            raise ValueError(f"{self.method} requires a frequency")
        if self.method in (AmortMethod.PERCENT_OF_ORIGINAL, AmortMethod.PERCENT_OF_CURRENT) and self.rate is None:
            raise ValueError(f"{self.method} requires a rate")
        if self.method is AmortMethod.FIXED_AMOUNT and self.amount is None:
            raise ValueError(f"{self.method} requires an amount")

    def payment_amount(self, opening_balance: float, original_balance: float) -> float:
        """Scheduled principal payment for one payment date within this rule.

        Does not clamp to the opening balance — callers (``resolve()``) are
        responsible for clamping so a payment never overdraws the facility.
        """
        if self.method is AmortMethod.PERCENT_OF_ORIGINAL:
            return self.rate * original_balance
        if self.method is AmortMethod.PERCENT_OF_CURRENT:
            return self.rate * opening_balance
        if self.method is AmortMethod.FIXED_AMOUNT:
            return self.amount
        if self.method is AmortMethod.STRAIGHT_LINE:
            raise NotImplementedError("STRAIGHT_LINE payment amount depends on remaining payment count; computed in schedule.py")
        if self.method is AmortMethod.BULLET:
            return 0.0
        raise AssertionError(f"unhandled method {self.method}")
