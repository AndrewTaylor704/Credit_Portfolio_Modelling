from datetime import datetime

import pytest

from credit_portfolio.models import DownturnLGD, MacroConditionedPD, RegulatoryCCF, Scenario, StaticPD
from credit_portfolio.valuation import ead, ecl, risk_weight, rwa
from tests.conftest import make_facility


def test_ead_is_facility_limit():
    facility = make_facility()
    assert ead(facility) == facility.limit


def test_ecl_matches_lgd_times_ead_times_pd():
    facility = make_facility()
    assert ecl(facility) == pytest.approx(facility.lgd * facility.limit * facility.customer.probdef)


def test_rwa_is_deterministic_for_a_fixed_as_of_date():
    facility = make_facility()
    as_of = datetime(2022, 1, 1)
    assert rwa(facility, as_of_date=as_of) == rwa(facility, as_of_date=as_of)


def test_rwa_varies_with_as_of_date_because_time_to_maturity_shortens():
    # This is the whole point of decoupling valuation from construction: the
    # same Facility object must be revaluable as of different dates.
    facility = make_facility()
    early = rwa(facility, as_of_date=datetime(2020, 6, 1))
    late = rwa(facility, as_of_date=datetime(2025, 6, 1))
    assert early != pytest.approx(late)


def test_risk_weight_clamps_effective_maturity_to_five_years():
    facility = make_facility()
    # Both dates are more than 5 years from the 2027-01-01 maturity, so the
    # effective maturity used in the formula is clamped to 5 either way.
    far_1 = risk_weight(facility, as_of_date=datetime(2020, 1, 2))
    far_2 = risk_weight(facility, as_of_date=datetime(2020, 6, 1))
    assert far_1 == pytest.approx(far_2)


def test_risk_weight_rejects_a_non_ttc_pd_model():
    facility = make_facility()
    with pytest.raises(ValueError, match="through-the-cycle"):
        risk_weight(facility, pd_model=MacroConditionedPD())


def test_rwa_rejects_a_non_ttc_pd_model():
    facility = make_facility()
    with pytest.raises(ValueError, match="through-the-cycle"):
        rwa(facility, pd_model=MacroConditionedPD())


def test_ecl_accepts_a_non_ttc_pd_model_and_a_scenario():
    facility = make_facility()
    facility.customer.sic_code = "1234"
    facility.customer.country = "GB"
    scenario = Scenario(z_scores={("1234", "GB"): -2.0})  # stressed conditions -> higher PD
    base = ecl(facility, pd_model=StaticPD())
    stressed = ecl(facility, pd_model=MacroConditionedPD(), scenario=scenario)
    assert stressed > base


def test_custom_lgd_and_ccf_models_reach_every_internal_call():
    facility = make_facility(type="Revolver", drawn_balance=400_000.0, limit=1_000_000.0)
    default_rwa = rwa(facility)
    custom_rwa = rwa(facility, lgd_model=DownturnLGD(addon=0.2), ccf_model=RegulatoryCCF())
    assert custom_rwa != pytest.approx(default_rwa)

    default_ecl = ecl(facility)
    custom_ecl = ecl(facility, lgd_model=DownturnLGD(addon=0.2), ccf_model=RegulatoryCCF())
    assert custom_ecl != pytest.approx(default_ecl)
