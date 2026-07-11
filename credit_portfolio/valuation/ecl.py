"""Expected credit loss.

``ecl()`` is a plain 1-year (12-month) expected loss -- LGD x EAD x PD --
used for RAROC/pricing purposes, where a 1-year economic view is what's
wanted regardless of accounting stage. Unlike ``risk_weight``/``rwa``, it
has no through-the-cycle constraint on ``pd_model`` -- a scenario-
conditioned ``MacroConditionedPD`` (or a ``DownturnLGD``) is a perfectly
valid input here.

``ifrs9_ecl()`` is the accounting-provisioning figure IFRS 9 actually
requires, and it is stage-dependent: Stage 1 gets the same 12-month EL as
``ecl()``; Stage 2 gets a lifetime EL (LGD x EAD x cumulative default
probability over the facility's remaining life, via the flat-hazard
assumption in ``models.pd.cumulative_default_probability``); Stage 3
(already credit-impaired) gets LGD x EAD directly (PD is taken as 1 -- the
facility has already defaulted). These are two different numbers serving
two different purposes -- do not use ``ecl()`` where an IFRS 9 provision
figure is required, or vice versa for pricing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from credit_portfolio.domain.facility import Facility
from credit_portfolio.models import StaticCCF, StaticLGD, StaticPD, cumulative_default_probability

from .ead import ead as _ead

_DEFAULT_PD_MODEL = StaticPD()
_DEFAULT_LGD_MODEL = StaticLGD()
_DEFAULT_CCF_MODEL = StaticCCF()

_VALID_IFRS_STAGES = (1, 2, 3)


def ecl(
    facility: Facility,
    as_of_date: datetime | None = None,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    lgd = lgd_model.lgd(facility, as_of_date, scenario)
    ead_value = _ead(facility, as_of_date, scenario, ccf_model=ccf_model)
    pd = pd_model.pd(facility.customer, as_of_date, scenario)
    return lgd * ead_value * pd


def ifrs9_ecl(
    facility: Facility,
    as_of_date: datetime,
    scenario: Any | None = None,
    pd_model=_DEFAULT_PD_MODEL,
    lgd_model=_DEFAULT_LGD_MODEL,
    ccf_model=_DEFAULT_CCF_MODEL,
) -> float:
    if facility.ifrs_stage not in _VALID_IFRS_STAGES:
        raise ValueError(f"facility {facility.facid!r} has an unrecognised IFRS_Stage {facility.ifrs_stage!r}; expected one of {_VALID_IFRS_STAGES}")

    lgd = lgd_model.lgd(facility, as_of_date, scenario)
    ead_value = _ead(facility, as_of_date, scenario, ccf_model=ccf_model)

    if facility.ifrs_stage == 3:
        # Already credit-impaired: PD is taken as 1, not re-derived from pd_model.
        return lgd * ead_value

    annual_pd = pd_model.pd(facility.customer, as_of_date, scenario)
    if facility.ifrs_stage == 1:
        return lgd * ead_value * annual_pd

    # Stage 2: lifetime ECL over the facility's remaining life.
    remaining_years = max((facility.maturity_date - as_of_date).days / 365.25, 0.0)
    lifetime_pd = cumulative_default_probability(annual_pd, remaining_years)
    return lgd * ead_value * lifetime_pd
