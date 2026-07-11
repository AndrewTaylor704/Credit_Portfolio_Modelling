"""Portfolio-level cashflow aggregation -- no-default path only.

Simulating which customers default and when, across a whole portfolio,
needs the Monte Carlo engine to retain per-scenario default draws (a known
gap in ``credit_portfolio.risk.montecarlo``, which currently only returns a
binned loss histogram) -- that integration is future work, not this
checkpoint.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from credit_portfolio.domain.currency import require_single_currency
from credit_portfolio.domain.portfolio import Portfolio
from credit_portfolio.models import StaticLGD

from .project import project_customer_cashflows
from .schedule import CashflowSchedule, merge_cashflow_schedules

_DEFAULT_LGD_MODEL = StaticLGD()


def portfolio_cashflows(
    portfolio: Portfolio,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    lgd_model=_DEFAULT_LGD_MODEL,
    funding_model=None,
    opex_model=None,
) -> CashflowSchedule:
    require_single_currency(f for c in portfolio.customer_list for f in c.facility_list)
    per_customer = [
        project_customer_cashflows(c, as_of_date, scenario, lgd_model, funding_model=funding_model, opex_model=opex_model)
        for c in portfolio.customer_list
    ]
    return merge_cashflow_schedules(per_customer)
