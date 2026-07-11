"""Maps a portfolio's customers to the broad sector/region buckets in
``live_data.py`` and builds a Scenario populated with each bucket's
live-computed Z-score, keyed by the customer's own (sic_code, country) so it
matches ``MacroConditionedPD.pd``'s lookup directly.

The sector mapping is a coarse SIC-code-prefix heuristic (not a full SIC/UK
SIC-2007 section table) -- customers whose (sic_code, country) doesn't map
to a covered bucket are simply left out of the returned Scenario, which
means they fall back to Z=0 (the base/TTC case) via ``Scenario.z_score``'s
existing ``.get(key, 0.0)``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from credit_portfolio.domain.customer import Customer
from credit_portfolio.models import Scenario

from .live_data import LIVE_DATA_POINTS
from .zscore import compute_zscore

_EU_COUNTRIES = {"DE", "FR", "IT", "ES", "NL", "EU"}


def _bucket_for(sic_code: str, country: str) -> Optional[str]:
    if country == "US":
        try:
            division = int(str(sic_code)[:2])
        except ValueError:
            return None
        if 5 <= division <= 33:  # coarse mining/manufacturing band
            return "US_MANUFACTURING"
        return None
    if country == "GB":
        return "UK_BROAD"
    if country in _EU_COUNTRIES:
        return "EU_BROAD"
    return None


def build_live_scenario(customers: Iterable[Customer]) -> Scenario:
    z_scores = {}
    for customer in customers:
        bucket_name = _bucket_for(customer.sic_code, customer.country)
        if bucket_name is None:
            continue
        point = LIVE_DATA_POINTS[bucket_name]
        z_scores[(customer.sic_code, customer.country)] = compute_zscore(
            point.current, point.historical_mean, point.historical_std, point.direction
        )
    return Scenario(z_scores=z_scores)
