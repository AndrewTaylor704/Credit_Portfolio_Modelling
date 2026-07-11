"""Sourced macro/equity data points, fetched via WebSearch on 2026-07-11.

This is a DEMONSTRATION dataset with real, current, cited numbers for a
handful of representative (sector, region) buckets -- it does not cover
every (sic_code, country) combination a real portfolio might contain (see
``scenario.py`` for how uncovered combinations fall back to Z=0, the
base/TTC case). Historical mean/std figures are approximations built from
the cited sources' own commentary (a commonly-quoted long-run average, or a
short run of recently-cited figures), not a computed statistic over a full
historical series -- this environment has no stable bulk time-series API
connector (see the design doc). Refreshing or extending this dataset means
re-running the same kind of search and updating the entries below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveDataPoint:
    bucket: str
    current: float
    historical_mean: float
    historical_std: float
    direction: float  # +1 if a higher reading is more benign, -1 if more stressed
    source: str
    as_of: str


LIVE_DATA_POINTS: dict[str, LiveDataPoint] = {
    "US_MANUFACTURING": LiveDataPoint(
        bucket="US_MANUFACTURING",
        current=53.3,
        historical_mean=52.5,
        historical_std=3.5,
        direction=1.0,
        source=(
            "ISM Manufacturing PMI Report, June 2026: 'Manufacturing PMI at 53.3%' "
            "(prnewswire.com/news-releases/manufacturing-pmi-at-53-3-june-2026-ism-manufacturing-pmi-report-302814991.html). "
            "Historical mean/std are the commonly-cited approximate long-run average/volatility for the series "
            "(expansion/contraction threshold = 50), not from that single article."
        ),
        as_of="2026-06 (released 2026-07-07)",
    ),
    "UK_BROAD": LiveDataPoint(
        bucket="UK_BROAD",
        current=4.9,
        historical_mean=4.5,
        historical_std=0.6,
        direction=-1.0,  # higher unemployment = more stressed
        source=(
            "ONS 'Employment in the UK: June 2026', unemployment rate 4.9% (three months to April 2026), "
            "via tradingeconomics.com/united-kingdom/unemployment-rate. Historical mean/std approximated from the "
            "cited 2023-2025 UK unemployment figures (4.03%, 4.36%, 4.75%) as a short recent baseline, not a "
            "computed 10-year statistic."
        ),
        as_of="2026-04 (three months to)",
    ),
    "EU_BROAD": LiveDataPoint(
        bucket="EU_BROAD",
        current=0.8,
        historical_mean=1.5,
        historical_std=1.0,
        direction=1.0,  # higher GDP growth = more benign
        source=(
            "Vanguard euro-area 2026 GDP growth forecast, 0.8% (corporate.vanguard.com/content/corporatesite/us/en/corp/vemo/vemo-europe.html) "
            "and ECB Eurosystem staff macroeconomic projections, June 2026 "
            "(ecb.europa.eu/press/projections/html/ecb.projections202606_eurosystemstaff~a495110f8d.en.html). "
            "Trend growth rate (~1.5%) as cited in the same market commentary."
        ),
        as_of="2026-Q1 (annualised forecast)",
    ),
}
