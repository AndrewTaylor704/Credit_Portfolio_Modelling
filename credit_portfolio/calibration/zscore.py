"""Converts a current indicator reading into a Z-score against a historical
baseline, for use as ``Scenario.z_scores`` input to ``MacroConditionedPD``."""

from __future__ import annotations


def compute_zscore(current: float, historical_mean: float, historical_std: float, direction: float = 1.0) -> float:
    """``direction`` is +1 if a higher reading is more benign (e.g. GDP
    growth, a PMI) or -1 if a higher reading is more stressed (e.g. an
    unemployment rate) -- it just flips the sign so the result always
    follows this codebase's convention (positive Z = benign conditions,
    matching ``risk.montecarlo``'s ``cust_asset_value``)."""
    if historical_std == 0:
        raise ValueError("historical_std must be non-zero")
    return direction * (current - historical_mean) / historical_std
