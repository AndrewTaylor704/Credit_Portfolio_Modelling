import pytest

from credit_portfolio.models import MacroConditionedPD, Scenario, StaticPD
from tests.conftest import make_customer


def test_static_pd_returns_probdef_unchanged():
    customer = make_customer(probdef=0.02)
    assert StaticPD().pd(customer) == 0.02


def test_static_pd_is_ttc():
    assert StaticPD.is_ttc is True


def test_macro_conditioned_pd_is_not_ttc():
    assert MacroConditionedPD.is_ttc is False


def test_macro_conditioned_pd_returns_ttc_pd_when_scenario_has_no_matching_key():
    customer = make_customer(probdef=0.02, sic_code="1234", country="GB")
    scenario = Scenario(z_scores={("9999", "US"): -3.0})  # no entry for this customer's key
    assert MacroConditionedPD().pd(customer, scenario=scenario) == pytest.approx(0.02)


def test_macro_conditioned_pd_returns_ttc_pd_at_zero_shock():
    customer = make_customer(probdef=0.02, sic_code="1234", country="GB")
    scenario = Scenario(z_scores={("1234", "GB"): 0.0})
    assert MacroConditionedPD().pd(customer, scenario=scenario) == pytest.approx(0.02)


def test_macro_conditioned_pd_rises_under_a_negative_shock():
    customer = make_customer(probdef=0.02, sic_code="1234", country="GB")
    scenario = Scenario(z_scores={("1234", "GB"): -2.0})
    assert MacroConditionedPD().pd(customer, scenario=scenario) > 0.02


def test_macro_conditioned_pd_falls_under_a_positive_shock():
    customer = make_customer(probdef=0.02, sic_code="1234", country="GB")
    scenario = Scenario(z_scores={("1234", "GB"): 2.0})
    assert MacroConditionedPD().pd(customer, scenario=scenario) < 0.02
