"""Retail IRB risk-weight formulas.

Unlike the wholesale (corporate/sovereign/bank/SME) formula, Basel's retail
IRB formulas use a fixed or PD-dependent asset correlation and have no
maturity adjustment term.

The correlation for each retail class is factored into its own function
(rather than inlined in the risk-weight formula) so ``raroc.economic_capital``
can reuse the exact same correlation its risk-weight formula uses.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def retail_mortgage_correlation() -> float:
    return 0.15


def retail_revolving_correlation() -> float:
    return 0.04


def retail_other_correlation(pd: float) -> float:
    return 0.03 + 0.16 * np.exp(-35 * pd)


def _retail_k(pd: float, lgd: float, R: float) -> float:
    return (lgd * norm.cdf(norm.ppf(pd) / np.sqrt(1 - R) + np.sqrt(R / (1 - R)) * norm.ppf(0.999))) - (lgd * pd)


def retail_mortgage_risk_weight(pd: float, lgd: float) -> float:
    return _retail_k(pd, lgd, R=retail_mortgage_correlation()) * 12.5


def retail_revolving_risk_weight(pd: float, lgd: float) -> float:
    return _retail_k(pd, lgd, R=retail_revolving_correlation()) * 12.5


def retail_other_risk_weight(pd: float, lgd: float) -> float:
    return _retail_k(pd, lgd, R=retail_other_correlation(pd)) * 12.5
