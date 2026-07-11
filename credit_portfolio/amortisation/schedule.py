"""Resolves an ``AmortisationSchedule`` template into a concrete, dated
payment table for a specific facility's start date, maturity date and
drawn balance.

Design invariants (see the design doc for the reasoning):
  - Every rule's active window is ``(window_start, window_end]`` — a
    payment posts at each sub-period's end, never at a window's start.
    This makes adjacent rules tile with no seam double-count or gap.
  - Rule windows must exactly partition ``[start_date, maturity_date]``;
    gaps and overlaps are hard errors, not silently ignored.
  - The final payment (always dated at ``maturity_date``) is an
    unconditional balance sweep: whatever is still owed is paid off
    there, regardless of what any rule's formula would have computed.
    Under-amortisation before maturity is the normal case (e.g. "1.5%/yr
    for the last 2 years, then bullet"), not an error. Over-amortisation
    (a rule computing a payment larger than the balance owed) raises a
    ``RuntimeWarning`` and clamps to the available balance instead.
"""

from __future__ import annotations

import bisect
import warnings
from dataclasses import dataclass, replace
from datetime import date
from typing import Sequence

from dateutil.relativedelta import relativedelta

from .rules import AmortMethod, AmortRule, RelativeAnchor

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AmortPayment:
    date: date
    opening_balance: float
    principal_payment: float
    closing_balance: float


@dataclass(frozen=True)
class AmortSchedule:
    """Resolved, dated payment table for one facility."""

    payments: tuple[AmortPayment, ...]

    def balance_on_date(self, as_of_date: date) -> float:
        """Outstanding balance as of ``as_of_date``.

        Before the first payment date this is the original drawn balance;
        on or after a payment date it's that payment's closing balance;
        after the final (maturity) payment it's 0.
        """
        if not self.payments:
            raise ValueError("schedule has no payments")
        dates = [p.date for p in self.payments]
        idx = bisect.bisect_right(dates, as_of_date) - 1
        if idx < 0:
            return self.payments[0].opening_balance
        return self.payments[idx].closing_balance


@dataclass(frozen=True)
class AmortisationSchedule:
    """Template: an ordered set of rule segments, anchored relative to a
    facility's start/maturity dates rather than to absolute dates."""

    rules: tuple[AmortRule, ...]

    @staticmethod
    def bullet() -> "AmortisationSchedule":
        return AmortisationSchedule(rules=(
            AmortRule(
                method=AmortMethod.BULLET,
                window_start=RelativeAnchor.from_start(),
                window_end=RelativeAnchor.from_maturity(),
            ),
        ))

    @staticmethod
    def straight_line(frequency) -> "AmortisationSchedule":
        from .rules import Frequency  # local import avoids a hard dependency for callers who only need bullet()
        assert isinstance(frequency, Frequency)
        return AmortisationSchedule(rules=(
            AmortRule(
                method=AmortMethod.STRAIGHT_LINE,
                window_start=RelativeAnchor.from_start(),
                window_end=RelativeAnchor.from_maturity(),
                frequency=frequency,
            ),
        ))

    @staticmethod
    def from_rules(rules: Sequence[AmortRule]) -> "AmortisationSchedule":
        """Build from explicit rules, each with its own window already set.

        Use this when a rule's window needs to be anchored to maturity
        (e.g. "the last 2 years") rather than chained from the start —
        ``segments()`` only supports the latter.
        """
        for rule in rules:
            if rule.window_start is None or rule.window_end is None:
                raise ValueError(f"rule {rule} is missing window_start/window_end")
        return AmortisationSchedule(rules=tuple(rules))

    @staticmethod
    def segments(parts: Sequence[tuple["relativedelta | None", AmortRule]]) -> "AmortisationSchedule":
        """Ergonomic constructor: each part is ``(duration, rule)``, chained
        so segment i's window starts where segment i-1's ends (segment 0
        starts at facility start). The last part's duration must be
        ``None``, meaning "run to maturity" — gaps/overlaps are therefore
        structurally impossible, unlike ``from_rules``.
        """
        if not parts:
            raise ValueError("segments() requires at least one part")
        resolved = []
        cumulative = relativedelta()
        for i, (duration, rule) in enumerate(parts):
            is_last = i == len(parts) - 1
            window_start = RelativeAnchor.from_start(cumulative)
            if duration is None:
                if not is_last:
                    raise ValueError("only the last segment may have an open-ended (None) duration")
                window_end = RelativeAnchor.from_maturity()
            else:
                if is_last:
                    raise ValueError("the last segment must have an open-ended (None) duration")
                cumulative = cumulative + duration
                window_end = RelativeAnchor.from_start(cumulative)
            resolved.append(replace(rule, window_start=window_start, window_end=window_end))
        return AmortisationSchedule(rules=tuple(resolved))

    def validate(self, start_date: date, maturity_date: date) -> None:
        """Raises ``ValueError`` unless the rule windows exactly partition
        ``[start_date, maturity_date]`` with no gap or overlap."""
        if not self.rules:
            raise ValueError("schedule has no rules")

        resolved = []
        for rule in self.rules:
            if rule.window_start is None or rule.window_end is None:
                raise ValueError(f"rule {rule} is missing window_start/window_end")
            w_start = rule.window_start.resolve(start_date, maturity_date)
            w_end = rule.window_end.resolve(start_date, maturity_date)
            if w_end <= w_start:
                raise ValueError(f"rule {rule} has a non-positive-length window: {w_start} -> {w_end}")
            resolved.append((w_start, w_end, rule))

        resolved.sort(key=lambda r: r[0])

        if resolved[0][0] != start_date:
            raise ValueError(
                f"rule windows must start at the facility start date {start_date}, "
                f"but the earliest window starts at {resolved[0][0]}"
            )
        if resolved[-1][1] != maturity_date:
            raise ValueError(
                f"rule windows must end at the facility maturity date {maturity_date}, "
                f"but the latest window ends at {resolved[-1][1]}"
            )
        for (_, end_a, rule_a), (start_b, _, rule_b) in zip(resolved, resolved[1:]):
            if end_a < start_b:
                raise ValueError(f"gap in amortisation schedule between {rule_a} (ends {end_a}) and {rule_b} (starts {start_b})")
            if end_a > start_b:
                raise ValueError(f"overlap in amortisation schedule between {rule_a} (ends {end_a}) and {rule_b} (starts {start_b})")


def _rule_payment_dates(rule: AmortRule, start_date: date, maturity_date: date) -> list[date]:
    if rule.method is AmortMethod.BULLET:
        return []
    window_start = rule.window_start.resolve(start_date, maturity_date)
    window_end = rule.window_end.resolve(start_date, maturity_date)
    step = rule.frequency.step
    dates: list[date] = []
    current = window_end
    while current > window_start:
        dates.append(current)
        current = current - step
    dates.reverse()
    return dates


def resolve(schedule: AmortisationSchedule, start_date: date, maturity_date: date, drawn_balance: float) -> AmortSchedule:
    """Pure function: (schedule template, facility dates/balance) -> a
    concrete dated payment table. No wall-clock date is ever consulted."""
    schedule.validate(start_date, maturity_date)

    ordered_rules = sorted(schedule.rules, key=lambda r: r.window_start.resolve(start_date, maturity_date))

    entries: list[tuple[date, AmortRule | None, int]] = []
    for rule in ordered_rules:
        dates = _rule_payment_dates(rule, start_date, maturity_date)
        for d in dates:
            entries.append((d, rule, len(dates)))

    if not entries or entries[-1][0] != maturity_date:
        entries.append((maturity_date, None, 0))

    opening_balance = drawn_balance
    original_balance = drawn_balance
    straight_line_rule_start_balance: dict[int, float] = {}
    payments: list[AmortPayment] = []
    last_index = len(entries) - 1

    for i, (d, rule, group_size) in enumerate(entries):
        is_final = i == last_index

        if rule is None:
            raw_amount = opening_balance
        elif rule.method is AmortMethod.STRAIGHT_LINE:
            key = id(rule)
            if key not in straight_line_rule_start_balance:
                straight_line_rule_start_balance[key] = opening_balance
            raw_amount = straight_line_rule_start_balance[key] / group_size
        else:
            raw_amount = rule.payment_amount(opening_balance, original_balance)

        if rule is not None and raw_amount > opening_balance + _TOLERANCE:
            warnings.warn(
                f"Amortisation rule {rule.method} on {d} would pay {raw_amount:.2f} against an "
                f"opening balance of {opening_balance:.2f}; rules over-amortise the facility "
                f"before maturity. Clamping to the available balance.",
                RuntimeWarning,
                stacklevel=2,
            )

        payment = opening_balance if is_final else min(raw_amount, opening_balance)
        closing_balance = max(opening_balance - payment, 0.0)
        payments.append(AmortPayment(date=d, opening_balance=opening_balance, principal_payment=payment, closing_balance=closing_balance))
        opening_balance = closing_balance

    return AmortSchedule(payments=tuple(payments))
