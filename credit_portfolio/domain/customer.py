"""Customer: a pure data holder. Aggregate valuation (ECL/RWA/EAD across a
customer's facilities) lives in ``credit_portfolio.valuation.aggregation``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from credit_portfolio.models.exposure_class import ExposureClass

if TYPE_CHECKING:
    from .facility import Facility


@dataclass
class Customer:
    customerid: str
    probdef: float
    sic_code: str
    country: str
    name: str
    turnover: float
    parent: Optional[str] = None
    facility_list: list["Facility"] = field(default_factory=list)
    exposure_class: ExposureClass = ExposureClass.CORPORATE
    external_rating: Optional[str] = None

    def __post_init__(self) -> None:
        self.turnover = self.turnover / 1_000_000

    def add_facility(self, facility: "Facility") -> None:
        self.facility_list.append(facility)
