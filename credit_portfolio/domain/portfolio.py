"""Portfolio: a pure data holder. Aggregate valuation lives in
``credit_portfolio.valuation.aggregation``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .customer import Customer


@dataclass
class Portfolio:
    id: str
    name: str
    customer_list: list["Customer"] = field(default_factory=list)

    @property
    def num_customers(self) -> int:
        return len(self.customer_list)

    def add_customer(self, customer: "Customer") -> None:
        self.customer_list.append(customer)
