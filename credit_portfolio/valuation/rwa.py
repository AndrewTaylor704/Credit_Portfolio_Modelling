"""IRB risk-weighted assets, dispatched by the customer's exposure class.

``corporate_irb_risk_weight`` (Basel's "corporate, sovereign, bank" wholesale
formula) is shared by ``ExposureClass.CORPORATE``/``SOVEREIGN``/``BANK``/
``SME`` — SME's size-based correlation discount already lives in
``asset_correlation`` via the turnover term. Retail exposure classes use
their own formulas (``rwa_retail.py``, no maturity adjustment); specialised
lending uses categorical supervisory slotting (``rwa_specialised_lending.py``),
not a PD/LGD formula at all.

Only the standardised approach (``rwa_standardised/``) is a real alternative
to IRB today — a genuinely different regulatory approach, not a gap.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from scipy.stats import norm

from credit_portfolio.domain.facility import Facility
from credit_portfolio.models import ExposureClass, WHOLESALE_CLASSES, StaticCCF, StaticLGD, StaticPD
from credit_portfolio.models.pd import asset_correlation

from .ead import ead as _ead
from .rwa_retail import retail_mortgage_risk_weight, retail_other_risk_weight, retail_revolving_risk_weight
from .rwa_specialised_lending import specialised_lending_risk_weight

_DEFAULT_PD_MODEL = StaticPD()
_DEFAULT_LGD_MODEL = StaticLGD()
_DEFAULT_CCF_MODEL = StaticCCF()


def corporate_irb_risk_weight(pd: float, lgd: float, maturity_years: float, turnover: float) -> float:
    """Basel wholesale (corporate/sovereign/bank/SME) IRB risk-weight formula."""
    effmat = max(min(maturity_years, 5), 1)
    R = asset_correlation(pd, turnover)
    b = (0.11852 - (0.05478 * np.log(pd))) ** 2
    matadj = (1 + ((effmat - 2.5) * b)) / (1 - (1.5 * b))
    K = (lgd * norm.cdf(norm.ppf(pd) / np.sqrt(1 - R) + np.sqrt(R / (1 - R)) * norm.ppf(0.999)) - (lgd * pd)) * matadj
    return K * 12.5


def _maturity_years(facility: Facility, as_of_date: datetime) -> float:
    return (facility.maturity_date - as_of_date).days / 365.25


def risk_weight(
    facility: Facility,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
) -> float:
    if not pd_model.is_ttc:
        raise ValueError(
            "risk_weight()/rwa() require a through-the-cycle PD model (pd_model.is_ttc == True); "
            "a point-in-time/macro-conditioned PD would double-count systematic risk against the "
            "IRB formula's own stress calibration. Use that pd_model with ecl() instead."
        )

    exposure_class = facility.customer.exposure_class
    if exposure_class is ExposureClass.SPECIALISED_LENDING:
        return specialised_lending_risk_weight(facility.supervisory_slot)

    as_of_date = as_of_date or datetime.now()
    pd = pd_model.pd(facility.customer, as_of_date, scenario)
    lgd = lgd_model.lgd(facility, as_of_date, scenario)

    if exposure_class in WHOLESALE_CLASSES:
        return corporate_irb_risk_weight(pd, lgd, _maturity_years(facility, as_of_date), facility.customer.turnover)
    if exposure_class is ExposureClass.RETAIL_MORTGAGE:
        return retail_mortgage_risk_weight(pd, lgd)
    if exposure_class is ExposureClass.RETAIL_REVOLVING:
        return retail_revolving_risk_weight(pd, lgd)
    if exposure_class is ExposureClass.RETAIL_OTHER:
        return retail_other_risk_weight(pd, lgd)
    raise AssertionError(f"unhandled exposure class {exposure_class}")


def rwa(
    facility: Facility,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    as_of_date = as_of_date or datetime.now()
    return _ead(facility, as_of_date, scenario, ccf_model=ccf_model) * risk_weight(
        facility, as_of_date, scenario, pd_model=pd_model, lgd_model=lgd_model
    )
