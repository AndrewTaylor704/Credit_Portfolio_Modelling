"""Currency-consistency guard for aggregation.

This codebase has no FX conversion capability. Summing monetary amounts
(EAD, ECL, RWA, cashflow, capital, ...) across facilities denominated in
different currencies without converting them first produces a number that
looks fine but is meaningless. ``require_single_currency`` turns that
silent wrong-number risk into a loud, actionable failure instead -- every
aggregation function in this codebase calls it before summing across
facilities.

Real multi-currency portfolios are entirely normal; the fix here is not to
forbid them, it's to require they be converted to one reporting currency
*before* reaching these aggregation functions (a real FX rate source is a
"needs real data" gap, not something this guard can paper over).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from .facility import Facility


def require_single_currency(facilities: Iterable["Facility"]) -> Optional[str]:
    currencies = {f.currency for f in facilities}
    if len(currencies) > 1:
        raise ValueError(
            f"cannot aggregate facilities in different currencies ({sorted(currencies)}) without FX "
            "conversion, which this codebase does not implement. Convert every facility to a single "
            "reporting currency before calling this aggregation, or aggregate each currency's facilities separately."
        )
    return next(iter(currencies), None)
