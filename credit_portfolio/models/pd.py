"""Probability-of-default models.

``asset_correlation`` lives here (not in ``valuation.rwa``) because both the
IRB wholesale risk-weight formula and ``MacroConditionedPD``'s PIT transform
need it, and this module has no dependency on ``valuation`` — keeping it
here avoids a circular import between the two.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.stats import norm

from .scenario import Scenario

if TYPE_CHECKING:
    from credit_portfolio.domain.customer import Customer


def asset_correlation(pd: float, turnover: float) -> float:
    """Basel's PD-dependent asset correlation, with the SME/turnover
    discount for turnover (in millions) at or below 50.

    This is a regulatory capital calibration constant (tuned so the IRB
    K-formula reproduces a supervisory view of 99.9th-percentile unexpected
    loss), not an empirically-estimated systematic asset correlation. Reusing
    it in ``MacroConditionedPD``'s PIT transform is a documented practitioner
    simplification, not a claim that it IS the true systematic correlation.
    """
    R = (0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))) + (
        0.24 * (1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)))
    )
    if turnover <= 50:
        R += -0.04 * (1 - ((max(turnover, 5) - 5) / 45))
    return R


def cumulative_default_probability(annual_pd: float, years: float) -> float:
    """Cumulative probability of default over ``years``, given a flat
    (constant-hazard) annual PD: ``1 - (1 - annual_pd) ** years``.

    Same flat-hazard simplification used in
    ``risk.cashflow_paths.simulate_cashflow_paths`` for multi-period default
    timing, and in ``valuation.ecl.ifrs9_ecl`` for lifetime (Stage 2) ECL --
    a real term structure of default probability isn't modelled.
    """
    years = max(years, 0.0)
    return 1 - (1 - annual_pd) ** years


class PDModel(Protocol):
    is_ttc: bool

    def pd(self, customer: "Customer", as_of_date: datetime | None, scenario: Scenario | None) -> float: ...


class StaticPD:
    """Returns customer.probdef unchanged — today's behaviour, treated as
    the through-the-cycle (TTC) PD."""

    is_ttc = True

    def pd(self, customer: "Customer", as_of_date: datetime | None = None, scenario: Scenario | None = None) -> float:
        return customer.probdef


class MacroConditionedPD:
    """Converts a TTC PD to a point-in-time PD via the single-factor
    Merton/Vasicek transform:

        PIT_pd = norm.cdf((norm.ppf(ttc_pd) - sqrt(R) * Y) / sqrt(1 - R))

    where ``Y`` is the scenario's Z-score for the customer's (sic_code,
    country) and ``R = asset_correlation(ttc_pd, customer.turnover)``. Sign
    convention matches ``risk.montecarlo``'s ``cust_asset_value``: a positive
    Z-score (benign conditions) lowers PD. At ``Y=0`` this returns the TTC PD
    unchanged.

    FOR ecl() / IFRS 9 STRESS USE ONLY. Basel's IRB formulas require a TTC
    PD — the systematic stress is already baked into their own
    correlation/quantile mechanics, so feeding them this PIT PD instead would
    double-count systematic risk. ``valuation.rwa.risk_weight``/``rwa`` guard
    against this via ``is_ttc``.
    """

    is_ttc = False

    def pd(self, customer: "Customer", as_of_date: datetime | None = None, scenario: Scenario | None = None) -> float:
        ttc_pd = customer.probdef
        y = scenario.z_score(customer.sic_code, customer.country) if scenario is not None else 0.0
        if y == 0.0:
            return ttc_pd
        R = asset_correlation(ttc_pd, customer.turnover)
        return norm.cdf((norm.ppf(ttc_pd) - np.sqrt(R) * y) / np.sqrt(1 - R))
