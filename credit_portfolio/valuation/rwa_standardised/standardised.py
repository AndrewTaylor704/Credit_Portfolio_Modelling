"""Standardised-approach RWA, as a genuine alternative to IRB. BCBS, EU
(CRR3), and UK (PRA Basel 3.1) jurisdictions are populated -- see
``tables.py`` for the researched figures and citations. Rated buckets are
shared across jurisdictions; EU/UK diverge only on unrated corporate/SME
treatment. Other jurisdiction-specific nuances (Sovereign/Bank/Retail/
Specialised-Lending, and the EU/UK SME supporting-factor capital discount)
remain a documented gap, falling back to the BCBS tables.
"""

from __future__ import annotations

from datetime import datetime

from credit_portfolio.domain.facility import Facility
from credit_portfolio.models import ExposureClass, StaticCCF

from ..ead import ead as _ead
from .tables import (
    BCBS_RATED_TABLES,
    EU_TRANSITIONAL_PD_THRESHOLD,
    EU_UNRATED_STANDARD_WEIGHT,
    EU_UNRATED_TRANSITIONAL_WEIGHT,
    MORTGAGE_FLAT_FALLBACK_WEIGHT,
    MORTGAGE_LTV_ABOVE_100_WEIGHT,
    MORTGAGE_LTV_BANDS,
    REGULATORY_RETAIL_WEIGHT,
    SPECIALISED_LENDING_PROJECT_FINANCE_PREOPERATIONAL_WEIGHT,
    SPECIALISED_LENDING_UNRATED_DEFAULT_WEIGHT,
    UK_UNRATED_IG_WEIGHT,
    UK_UNRATED_NON_IG_WEIGHT,
)

_DEFAULT_CCF_MODEL = StaticCCF()

_SUPPORTED_JURISDICTIONS = {"BCBS", "EU", "UK"}
_CORPORATE_LIKE_CLASSES = (ExposureClass.CORPORATE, ExposureClass.SME)


def _lookup_rating(table: dict[str, float], external_rating: str) -> float:
    if external_rating not in table:
        raise ValueError(f"unknown rating bucket {external_rating!r}; expected one of {sorted(table)}")
    return table[external_rating]


def _mortgage_weight(ltv: float | None) -> float:
    if ltv is None:
        return MORTGAGE_FLAT_FALLBACK_WEIGHT
    for max_ltv, weight in MORTGAGE_LTV_BANDS:
        if ltv <= max_ltv:
            return weight
    return MORTGAGE_LTV_ABOVE_100_WEIGHT


def _specialised_lending_unrated_weight(subtype: str | None, phase: str | None) -> float:
    if subtype == "project_finance" and phase == "pre_operational":
        return SPECIALISED_LENDING_PROJECT_FINANCE_PREOPERATIONAL_WEIGHT
    return SPECIALISED_LENDING_UNRATED_DEFAULT_WEIGHT


def _eu_unrated_corporate_weight(internal_pd: float | None) -> float:
    """CRR3 transitional treatment (expires 2032): 65% if the bank's own
    internal PD estimate for the obligor is <= 0.5%, else 100%. No PD
    evidence supplied -> the standard (non-preferential) weight."""
    if internal_pd is not None and internal_pd <= EU_TRANSITIONAL_PD_THRESHOLD:
        return EU_UNRATED_TRANSITIONAL_WEIGHT
    return EU_UNRATED_STANDARD_WEIGHT


def _uk_unrated_corporate_weight(is_investment_grade: bool | None) -> float:
    """PRA Basel 3.1 risk-sensitive split: 65% if the firm assesses the
    exposure as investment-grade, else 135% -- including when no assessment
    is supplied (conservative fallback, same reasoning as this codebase's
    required funding_model/opex_model in RAROC: an unassessed exposure
    should not silently get the favourable weight)."""
    if is_investment_grade is True:
        return UK_UNRATED_IG_WEIGHT
    return UK_UNRATED_NON_IG_WEIGHT


def standardised_risk_weight(
    jurisdiction: str,
    exposure_class: ExposureClass,
    external_rating: str | None = None,
    ltv: float | None = None,
    specialised_lending_subtype: str | None = None,
    specialised_lending_phase: str | None = None,
    internal_pd: float | None = None,
    is_investment_grade: bool | None = None,
) -> float:
    if jurisdiction not in _SUPPORTED_JURISDICTIONS:
        raise NotImplementedError(
            f"standardised approach for jurisdiction {jurisdiction!r} is not implemented yet "
            f"(only {sorted(_SUPPORTED_JURISDICTIONS)} are)"
        )

    if exposure_class is ExposureClass.RETAIL_MORTGAGE:
        return _mortgage_weight(ltv)
    if exposure_class in (ExposureClass.RETAIL_REVOLVING, ExposureClass.RETAIL_OTHER):
        return REGULATORY_RETAIL_WEIGHT
    if exposure_class is ExposureClass.SPECIALISED_LENDING:
        if external_rating is not None:
            corporate_table, _ = BCBS_RATED_TABLES[ExposureClass.CORPORATE]
            return _lookup_rating(corporate_table, external_rating)
        return _specialised_lending_unrated_weight(specialised_lending_subtype, specialised_lending_phase)

    table, unrated_weight = BCBS_RATED_TABLES[exposure_class]
    if external_rating is not None:
        return _lookup_rating(table, external_rating)

    if exposure_class in _CORPORATE_LIKE_CLASSES:
        if jurisdiction == "EU":
            return _eu_unrated_corporate_weight(internal_pd)
        if jurisdiction == "UK":
            return _uk_unrated_corporate_weight(is_investment_grade)
    return unrated_weight


def rwa_standardised(
    facility: Facility,
    as_of_date: datetime | None = None,
    jurisdiction: str = "BCBS",
    ccf_model=_DEFAULT_CCF_MODEL,
    internal_pd: float | None = None,
    is_investment_grade: bool | None = None,
) -> float:
    customer = facility.customer
    weight = standardised_risk_weight(
        jurisdiction,
        customer.exposure_class,
        external_rating=customer.external_rating,
        ltv=facility.ltv,
        specialised_lending_subtype=facility.specialised_lending_subtype,
        specialised_lending_phase=facility.specialised_lending_phase,
        internal_pd=internal_pd,
        is_investment_grade=is_investment_grade,
    )
    return _ead(facility, as_of_date, ccf_model=ccf_model) * weight
