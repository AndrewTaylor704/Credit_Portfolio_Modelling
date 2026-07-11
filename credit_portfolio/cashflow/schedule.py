"""Dated cashflow events and the schedule/merge machinery shared by
facility-, customer-, and portfolio-level cashflow projection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence


@dataclass(frozen=True)
class CashflowEvent:
    date: date
    interest: float = 0.0
    commitment_fee: float = 0.0
    upfront_fee: float = 0.0
    principal: float = 0.0
    recovery: float = 0.0
    funding_cost: float = 0.0
    operating_cost: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.interest
            + self.commitment_fee
            + self.upfront_fee
            + self.principal
            + self.recovery
            - self.funding_cost
            - self.operating_cost
        )


@dataclass(frozen=True)
class CashflowSchedule:
    events: tuple[CashflowEvent, ...]


def merge_cashflow_schedules(schedules: Sequence[CashflowSchedule]) -> CashflowSchedule:
    """Sums cashflow events across schedules by date, over the union of
    all dates that appear in any of them."""
    by_date: dict[date, list[float]] = {}
    for schedule in schedules:
        for event in schedule.events:
            totals = by_date.setdefault(event.date, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            totals[0] += event.interest
            totals[1] += event.commitment_fee
            totals[2] += event.upfront_fee
            totals[3] += event.principal
            totals[4] += event.recovery
            totals[5] += event.funding_cost
            totals[6] += event.operating_cost

    merged = tuple(
        CashflowEvent(
            date=d,
            interest=t[0],
            commitment_fee=t[1],
            upfront_fee=t[2],
            principal=t[3],
            recovery=t[4],
            funding_cost=t[5],
            operating_cost=t[6],
        )
        for d, t in sorted(by_date.items())
    )
    return CashflowSchedule(events=merged)
