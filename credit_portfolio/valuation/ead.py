"""Exposure at default.

Defaults to ``StaticCCF`` (today's behaviour: EAD = facility limit). Pass a
``ccf_model`` such as ``RegulatoryCCF`` to apply a credit-conversion factor
to the undrawn commitment instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from credit_portfolio.domain.facility import Facility
from credit_portfolio.models import StaticCCF

_DEFAULT_CCF_MODEL = StaticCCF()


def ead(
    facility: Facility,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    return ccf_model.exposure(facility, as_of_date, scenario)
