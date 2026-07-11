"""Basel exposure classes.

Corporate, Sovereign, Bank and SME share one IRB formula in Basel (the
"corporate, sovereign, bank" asset-class group) — SME gets its size-based
correlation discount from the turnover-based term already present in
``valuation.rwa.asset_correlation``, so it needs no separate IRB formula.
It does get separate (preferential) treatment under the standardised
approach, which is why it's a distinct exposure class rather than folded
into CORPORATE.
"""

from __future__ import annotations

from enum import Enum, auto


class ExposureClass(Enum):
    CORPORATE = auto()
    SOVEREIGN = auto()
    BANK = auto()
    SME = auto()
    RETAIL_MORTGAGE = auto()
    RETAIL_REVOLVING = auto()
    RETAIL_OTHER = auto()
    SPECIALISED_LENDING = auto()


WHOLESALE_CLASSES = frozenset({
    ExposureClass.CORPORATE,
    ExposureClass.SOVEREIGN,
    ExposureClass.BANK,
    ExposureClass.SME,
})
