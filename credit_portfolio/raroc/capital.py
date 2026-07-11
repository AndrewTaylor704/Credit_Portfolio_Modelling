"""Capital denominators for RAROC: regulatory (RWA-based) or a standalone
closed-form economic capital estimate."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from scipy.stats import norm

from credit_portfolio.domain.facility import Facility
from credit_portfolio.models import ExposureClass, StaticCCF, StaticLGD, StaticPD, WHOLESALE_CLASSES
from credit_portfolio.models.pd import asset_correlation
from credit_portfolio.valuation.ead import ead as _ead
from credit_portfolio.valuation.rwa import rwa
from credit_portfolio.valuation.rwa_retail import (
    retail_mortgage_correlation,
    retail_other_correlation,
    retail_revolving_correlation,
)

_DEFAULT_PD_MODEL = StaticPD()
_DEFAULT_LGD_MODEL = StaticLGD()
_DEFAULT_CCF_MODEL = StaticCCF()


def regulatory_capital(
    facility: Facility,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
    target_capital_ratio: float = 0.08,
) -> float:
    """target_capital_ratio defaults to the Pillar 1 minimum (8%) -- a
    bank's actual management target (often 12-15% including buffers) should
    be passed explicitly."""
    return rwa(facility, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model, ccf_model=ccf_model) * target_capital_ratio


def _correlation_for(exposure_class: ExposureClass, pd: float, turnover: float) -> float:
    if exposure_class in WHOLESALE_CLASSES:
        return asset_correlation(pd, turnover)
    if exposure_class is ExposureClass.RETAIL_MORTGAGE:
        return retail_mortgage_correlation()
    if exposure_class is ExposureClass.RETAIL_REVOLVING:
        return retail_revolving_correlation()
    if exposure_class is ExposureClass.RETAIL_OTHER:
        return retail_other_correlation(pd)
    raise AssertionError(f"unhandled exposure class {exposure_class}")


def economic_capital(
    facility: Facility,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
    confidence: float = 0.999,
) -> float:
    """Standalone (non-diversified) single-factor Vasicek unexpected loss,
    at a bank-chosen confidence level -- NOT true diversified portfolio
    economic capital (that needs Monte Carlo per-scenario allocation across
    correlated obligors, a flagged future gap). Requires a through-the-cycle
    pd_model (same double-counting guard as risk_weight()/rwa()) and is not
    defined for ExposureClass.SPECIALISED_LENDING (categorical supervisory
    slotting has no PD/LGD-formula correlation to reuse).
    """
    if not pd_model.is_ttc:
        raise ValueError("economic_capital() requires a through-the-cycle PD model, same as risk_weight()/rwa().")

    exposure_class = facility.customer.exposure_class
    if exposure_class is ExposureClass.SPECIALISED_LENDING:
        raise NotImplementedError(
            "economic_capital() is not defined for SPECIALISED_LENDING; use regulatory_capital() instead."
        )

    as_of_date = as_of_date or datetime.now()
    pd = pd_model.pd(facility.customer, as_of_date, scenario)
    lgd = lgd_model.lgd(facility, as_of_date, scenario)
    R = _correlation_for(exposure_class, pd, facility.customer.turnover)
    wcdr = norm.cdf((norm.ppf(pd) + np.sqrt(R) * norm.ppf(confidence)) / np.sqrt(1 - R))
    K = lgd * (wcdr - pd)
    return _ead(facility, as_of_date, scenario, ccf_model=ccf_model) * K
