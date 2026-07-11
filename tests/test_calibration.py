import math

import pytest

from credit_portfolio.calibration import build_live_scenario, compute_zscore
from tests.conftest import make_customer


def test_compute_zscore_matches_formula():
    assert compute_zscore(current=53.3, historical_mean=52.5, historical_std=3.5, direction=1.0) == pytest.approx(
        (53.3 - 52.5) / 3.5
    )


def test_compute_zscore_direction_flips_sign():
    positive_direction = compute_zscore(current=4.9, historical_mean=4.5, historical_std=0.6, direction=1.0)
    negative_direction = compute_zscore(current=4.9, historical_mean=4.5, historical_std=0.6, direction=-1.0)
    assert negative_direction == pytest.approx(-positive_direction)


def test_compute_zscore_rejects_zero_std():
    with pytest.raises(ValueError):
        compute_zscore(current=1.0, historical_mean=1.0, historical_std=0.0)


def test_build_live_scenario_covers_gb_customers_with_finite_zscores():
    customers = [make_customer(customerid="C1", sic_code="07210", country="GB")]
    scenario = build_live_scenario(customers)

    assert len(scenario.z_scores) == 1
    z = scenario.z_score("07210", "GB")
    assert math.isfinite(z)
    assert z != 0.0


def test_build_live_scenario_leaves_uncovered_customers_out_and_they_default_to_zero():
    customers = [make_customer(customerid="C1", sic_code="1234", country="ZZ")]
    scenario = build_live_scenario(customers)

    assert scenario.z_scores == {}
    assert scenario.z_score("1234", "ZZ") == 0.0
