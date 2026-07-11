"""Facility: a pure data holder for a single credit facility's static terms.

Deliberately holds no valuation fields (EAD/ECL/RWA/risk weight) and no
resolved amortisation payments — those depend on an as-of date (and, in
future, a scenario) and live in ``credit_portfolio.valuation`` /
``credit_portfolio.amortisation`` as explicit functions instead, so the same
Facility can be revalued repeatedly under different conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from credit_portfolio.amortisation import AmortisationSchedule, Frequency

if TYPE_CHECKING:
    from .customer import Customer


@dataclass
class Facility:
    facid: str
    lgd: float
    type: str
    start_date: datetime
    maturity_date: datetime
    limit: float
    drawn_balance: float
    margin: float
    fee: float
    currency: str
    ifrs_stage: int
    customerid: str
    customer: "Customer"
    amort_schedule: AmortisationSchedule = field(
        default_factory=lambda: AmortisationSchedule.straight_line(Frequency.QUARTERLY)
    )
    # Used only for ExposureClass.SPECIALISED_LENDING facilities:
    supervisory_slot: Optional[str] = None  # IRB supervisory slotting category
    specialised_lending_subtype: Optional[str] = None  # "project_finance" | "object_finance" | "commodities_finance" | "income_producing_real_estate"
    specialised_lending_phase: Optional[str] = None  # "pre_operational" | "operational"
    # Used only for ExposureClass.RETAIL_MORTGAGE facilities:
    ltv: Optional[float] = None
    # One-off arrangement fee rate on the committed limit, accrued into
    # income straight-line over the facility's life (not recognised
    # upfront). `fee` above is the recurring commitment-fee rate on the
    # undrawn commitment -- a separate concept.
    upfront_fee_rate: Optional[float] = None
