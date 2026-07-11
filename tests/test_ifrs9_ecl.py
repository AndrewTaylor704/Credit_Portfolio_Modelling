from datetime import datetime

import pytest

from credit_portfolio.models import cumulative_default_probability
from credit_portfolio.valuation import ecl, ifrs9_ecl
from tests.conftest import make_facility

AS_OF = datetime(2020, 1, 1)


def test_stage_1_matches_the_plain_12_month_ecl():
    facility = make_facility(ifrs_stage=1, lgd=0.45)
    facility.customer.probdef = 0.02
    assert ifrs9_ecl(facility, AS_OF) == pytest.approx(ecl(facility, AS_OF))


def test_stage_2_uses_lifetime_cumulative_default_probability():
    facility = make_facility(ifrs_stage=2, lgd=0.45, maturity_date=datetime(2025, 1, 1))
    facility.customer.probdef = 0.02

    remaining_years = (facility.maturity_date - AS_OF).days / 365.25
    expected_lifetime_pd = cumulative_default_probability(0.02, remaining_years)
    expected = facility.lgd * facility.limit * expected_lifetime_pd

    assert ifrs9_ecl(facility, AS_OF) == pytest.approx(expected)
    # Lifetime EL should exceed the 1-year EL for the same facility.
    assert ifrs9_ecl(facility, AS_OF) > ecl(facility, AS_OF)


def test_stage_3_is_lgd_times_ead_with_no_pd():
    facility = make_facility(ifrs_stage=3, lgd=0.45)
    facility.customer.probdef = 0.02
    assert ifrs9_ecl(facility, AS_OF) == pytest.approx(facility.lgd * facility.limit)


def test_unrecognised_stage_raises():
    facility = make_facility(ifrs_stage=99)
    with pytest.raises(ValueError, match="IFRS_Stage"):
        ifrs9_ecl(facility, AS_OF)
