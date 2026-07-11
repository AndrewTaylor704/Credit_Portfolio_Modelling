import pytest

from credit_portfolio.models import ExposureClass
from credit_portfolio.valuation import rwa_standardised, standardised_risk_weight
from tests.conftest import make_customer, make_facility


def test_unrated_corporate_gets_flat_100_percent():
    assert standardised_risk_weight("BCBS", ExposureClass.CORPORATE, external_rating=None) == pytest.approx(1.00)


def test_unrated_sme_gets_the_basel_iii_preferential_85_percent():
    # The key regression test for the Basel III change vs. Basel II's flat 100%.
    assert standardised_risk_weight("BCBS", ExposureClass.SME, external_rating=None) == pytest.approx(0.85)


def test_rated_corporate_bucket():
    assert standardised_risk_weight("BCBS", ExposureClass.CORPORATE, external_rating="A") == pytest.approx(0.50)


def test_rated_sovereign_bucket():
    assert standardised_risk_weight("BCBS", ExposureClass.SOVEREIGN, external_rating="AAA") == pytest.approx(0.0)


def test_mortgage_uses_ltv_band():
    assert standardised_risk_weight("BCBS", ExposureClass.RETAIL_MORTGAGE, ltv=0.55) == pytest.approx(0.25)


def test_mortgage_falls_back_to_flat_weight_without_ltv():
    assert standardised_risk_weight("BCBS", ExposureClass.RETAIL_MORTGAGE, ltv=None) == pytest.approx(0.35)


def test_project_finance_preoperational_gets_130_percent():
    weight = standardised_risk_weight(
        "BCBS",
        ExposureClass.SPECIALISED_LENDING,
        specialised_lending_subtype="project_finance",
        specialised_lending_phase="pre_operational",
    )
    assert weight == pytest.approx(1.30)


def test_project_finance_operational_gets_flat_100_percent():
    weight = standardised_risk_weight(
        "BCBS",
        ExposureClass.SPECIALISED_LENDING,
        specialised_lending_subtype="project_finance",
        specialised_lending_phase="operational",
    )
    assert weight == pytest.approx(1.00)


def test_unsupported_jurisdiction_raises():
    with pytest.raises(NotImplementedError):
        standardised_risk_weight("US", ExposureClass.CORPORATE, external_rating=None)


def test_eu_unrated_corporate_gets_transitional_65_percent_below_pd_threshold():
    weight = standardised_risk_weight("EU", ExposureClass.CORPORATE, external_rating=None, internal_pd=0.003)
    assert weight == pytest.approx(0.65)


def test_eu_unrated_corporate_gets_standard_100_percent_above_pd_threshold():
    weight = standardised_risk_weight("EU", ExposureClass.CORPORATE, external_rating=None, internal_pd=0.01)
    assert weight == pytest.approx(1.00)


def test_eu_unrated_corporate_gets_standard_100_percent_without_pd_evidence():
    weight = standardised_risk_weight("EU", ExposureClass.CORPORATE, external_rating=None, internal_pd=None)
    assert weight == pytest.approx(1.00)


def test_eu_rated_corporate_bucket_matches_bcbs():
    assert standardised_risk_weight("EU", ExposureClass.CORPORATE, external_rating="A") == pytest.approx(0.50)


def test_uk_unrated_corporate_investment_grade_gets_65_percent():
    weight = standardised_risk_weight("UK", ExposureClass.CORPORATE, external_rating=None, is_investment_grade=True)
    assert weight == pytest.approx(0.65)


def test_uk_unrated_corporate_non_investment_grade_gets_135_percent():
    weight = standardised_risk_weight("UK", ExposureClass.CORPORATE, external_rating=None, is_investment_grade=False)
    assert weight == pytest.approx(1.35)


def test_uk_unrated_corporate_without_assessment_falls_back_to_non_ig_135_percent():
    weight = standardised_risk_weight("UK", ExposureClass.CORPORATE, external_rating=None, is_investment_grade=None)
    assert weight == pytest.approx(1.35)


def test_uk_rated_corporate_bucket_matches_bcbs():
    assert standardised_risk_weight("UK", ExposureClass.CORPORATE, external_rating="BBB") == pytest.approx(0.75)


def test_rwa_standardised_composes_with_ead():
    customer = make_customer(exposure_class=ExposureClass.CORPORATE, external_rating="BBB")
    facility = make_facility(customer=customer, limit=1_000_000.0)
    assert rwa_standardised(facility) == pytest.approx(1_000_000.0 * 0.75)
