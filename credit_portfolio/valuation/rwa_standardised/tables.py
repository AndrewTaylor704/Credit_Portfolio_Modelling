"""BCBS Basel III (Dec 2017 finalisation) revised standardised-approach
risk-weight tables, plus the EU (CRR3) / UK (PRA Basel 3.1) unrated-
corporate divergences researched for checkpoint 7 — simplified. Rated
buckets are shared across all three jurisdictions (confirmed via research:
CRR3 and PRA Basel 3.1 both follow the BCBS rated-bucket structure); no
transactor-QRRE 45% preferential rate, no short-maturity specialised-
lending discount, and EU/UK Sovereign/Bank/Retail/Specialised-Lending
treatment falls back to the BCBS tables (a documented gap, not full
jurisdiction-specific granularity there).

Rating buckets are simplified to one label per notch-group (e.g. "AA"
covers AA+/AA/AA-), not full notch-level ratings.

EU/UK unrated-corporate figures, researched this session:
  - EU (CRR3, effective 1 Jan 2025, transitional to 2032): unrated
    corporates/SME get a TRANSITIONAL 65% risk weight if the bank's own
    internal PD estimate for that obligor is <= 0.5%, otherwise 100%. This
    transitional treatment expires in 2032 -- source: AFME "CRR3: A
    Risk-Sensitive and Proportional Approach to the Credit Risk Framework"
    (afme.eu) and Advisense "CRR3 - a More Level Playing Field Between
    Banks" (2025-03-27), both confirming the 65%/PD<=0.5%/2032-sunset
    structure.
  - UK (PRA Basel 3.1, effective 1 Jan 2027, full implementation by 2030):
    unrated corporates get 65% if the firm assesses the exposure as
    investment-grade, or 135% (not the BCBS/EU 100%) if non-investment-
    grade -- a deliberate UK divergence from BCBS/EU. Source: PRA policy
    statement PS1/26 (Bank of England, published 2026-01-20) as summarised
    by Grant Thornton "Basel 3.1: PRA clarifies a standardised approach to
    credit risk".
"""

from __future__ import annotations

from credit_portfolio.models import ExposureClass

_CORPORATE_TABLE = {"AAA": 0.20, "AA": 0.20, "A": 0.50, "BBB": 0.75, "BB": 1.00, "B": 1.00, "CCC": 1.50}
_SOVEREIGN_TABLE = {"AAA": 0.0, "AA": 0.0, "A": 0.20, "BBB": 0.50, "BB": 1.00, "B": 1.00, "CCC": 1.50}
_BANK_TABLE = {"AAA": 0.20, "AA": 0.20, "A": 0.30, "BBB": 0.50, "BB": 1.00, "B": 1.00, "CCC": 1.50}

# (rated table, unrated fallback weight) by exposure class.
BCBS_RATED_TABLES: dict[ExposureClass, tuple[dict[str, float], float]] = {
    ExposureClass.CORPORATE: (_CORPORATE_TABLE, 1.00),
    # SME: same rated buckets as Corporate, but 85% unrated -- the actual
    # Basel III change vs. Basel II's flat 100% unrated corporate treatment.
    ExposureClass.SME: (_CORPORATE_TABLE, 0.85),
    ExposureClass.SOVEREIGN: (_SOVEREIGN_TABLE, 1.00),
    # Simplified flat unrated fallback rather than full SCRA grading.
    ExposureClass.BANK: (_BANK_TABLE, 0.50),
}

# Retail mortgage: LTV-banded (checked in ascending order; first match wins).
MORTGAGE_LTV_BANDS: list[tuple[float, float]] = [
    (0.50, 0.20),
    (0.60, 0.25),
    (0.80, 0.30),
    (0.90, 0.40),
    (1.00, 0.50),
]
MORTGAGE_LTV_ABOVE_100_WEIGHT = 0.70
MORTGAGE_FLAT_FALLBACK_WEIGHT = 0.35  # used when ltv is unavailable (whole-loan approach)

REGULATORY_RETAIL_WEIGHT = 0.75  # retail revolving / retail other

# Specialised lending, unrated: rated exposures use the Corporate table instead.
SPECIALISED_LENDING_PROJECT_FINANCE_PREOPERATIONAL_WEIGHT = 1.30
SPECIALISED_LENDING_UNRATED_DEFAULT_WEIGHT = 1.00

# EU (CRR3): transitional unrated corporate/SME treatment, expires 2032.
EU_UNRATED_TRANSITIONAL_WEIGHT = 0.65
EU_UNRATED_STANDARD_WEIGHT = 1.00
EU_TRANSITIONAL_PD_THRESHOLD = 0.005

# UK (PRA Basel 3.1): risk-sensitive unrated corporate/SME split.
UK_UNRATED_IG_WEIGHT = 0.65
UK_UNRATED_NON_IG_WEIGHT = 1.35
