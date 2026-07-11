"""IRB supervisory slotting for specialised lending.

Basel lets banks without PD estimates for specialised lending use
supervisory-assigned categorical risk weights instead of a PD/LGD formula.
Simplified here: no preferential short-remaining-maturity discount (Basel
allows e.g. "Strong"/"Good" to drop to 50%/70% for remaining maturity under
2.5 years — not modelled).
"""

from __future__ import annotations

_SLOTTING_RISK_WEIGHTS = {
    "Strong": 0.70,
    "Good": 0.90,
    "Satisfactory": 1.15,
    "Weak": 2.50,
    "Default": 0.0,
}


def specialised_lending_risk_weight(supervisory_slot: str | None) -> float:
    if supervisory_slot not in _SLOTTING_RISK_WEIGHTS:
        raise ValueError(
            f"unknown supervisory slot {supervisory_slot!r}; expected one of {sorted(_SLOTTING_RISK_WEIGHTS)}"
        )
    return _SLOTTING_RISK_WEIGHTS[supervisory_slot]
