"""A scenario carries the inputs that condition risk parameters away from
their through-the-cycle base case.

``z_scores`` is keyed by ``(sic_code, country)`` — a systemic factor in
standard-normal units, positive meaning benign credit conditions relative to
that sector/region's mean, negative meaning stressed. A missing key defaults
to ``0.0`` (the base/TTC case).

Calibrating these Z-scores from real macro-economic and equity data (which
series, which universe per sector/region, how "mean" and volatility are
defined) is a separate, future piece of work — this is only the lookup shape
that calibration will populate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    z_scores: dict[tuple[str, str], float] = field(default_factory=dict)

    def z_score(self, sic_code: str, country: str) -> float:
        return self.z_scores.get((sic_code, country), 0.0)
