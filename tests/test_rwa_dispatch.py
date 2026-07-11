import pytest

from credit_portfolio.models import ExposureClass, StaticLGD, StaticPD
from credit_portfolio.valuation import (
    corporate_irb_risk_weight,
    retail_mortgage_risk_weight,
    retail_other_risk_weight,
    retail_revolving_risk_weight,
    risk_weight,
    specialised_lending_risk_weight,
)
from credit_portfolio.valuation.rwa import _maturity_years
from tests.conftest import make_customer, make_facility

AS_OF = None  # use "now" consistently via risk_weight's own default


class _ExplodingModel:
    """A model stub that raises if it's ever called -- used to prove a
    branch doesn't need pd_model/lgd_model at all."""

    is_ttc = True

    def pd(self, *args, **kwargs):
        raise AssertionError("pd_model should not be called for this exposure class")

    def lgd(self, *args, **kwargs):
        raise AssertionError("lgd_model should not be called for this exposure class")


def _wholesale_facility(exposure_class):
    customer = make_customer(exposure_class=exposure_class, probdef=0.01, turnover=30_000_000)
    return make_facility(customer=customer)


@pytest.mark.parametrize("exposure_class", [
    ExposureClass.CORPORATE, ExposureClass.SOVEREIGN, ExposureClass.BANK, ExposureClass.SME,
])
def test_wholesale_classes_route_to_corporate_irb_formula(exposure_class):
    facility = _wholesale_facility(exposure_class)
    result = risk_weight(facility)
    expected = corporate_irb_risk_weight(
        pd=facility.customer.probdef,
        lgd=facility.lgd,
        maturity_years=_maturity_years(facility, __import__("datetime").datetime.now()),
        turnover=facility.customer.turnover,
    )
    assert result == pytest.approx(expected, rel=1e-6)


def test_retail_mortgage_routes_to_retail_mortgage_formula():
    customer = make_customer(exposure_class=ExposureClass.RETAIL_MORTGAGE, probdef=0.01)
    facility = make_facility(customer=customer)
    assert risk_weight(facility) == pytest.approx(retail_mortgage_risk_weight(0.01, facility.lgd))


def test_retail_revolving_routes_to_retail_revolving_formula():
    customer = make_customer(exposure_class=ExposureClass.RETAIL_REVOLVING, probdef=0.01)
    facility = make_facility(customer=customer)
    assert risk_weight(facility) == pytest.approx(retail_revolving_risk_weight(0.01, facility.lgd))


def test_retail_other_routes_to_retail_other_formula():
    customer = make_customer(exposure_class=ExposureClass.RETAIL_OTHER, probdef=0.01)
    facility = make_facility(customer=customer)
    assert risk_weight(facility) == pytest.approx(retail_other_risk_weight(0.01, facility.lgd))


def test_specialised_lending_routes_to_supervisory_slotting():
    customer = make_customer(exposure_class=ExposureClass.SPECIALISED_LENDING)
    facility = make_facility(customer=customer, supervisory_slot="Good")
    assert risk_weight(facility) == pytest.approx(specialised_lending_risk_weight("Good"))


def test_specialised_lending_never_calls_pd_or_lgd_models():
    customer = make_customer(exposure_class=ExposureClass.SPECIALISED_LENDING)
    facility = make_facility(customer=customer, supervisory_slot="Strong")
    # Would raise AssertionError if either model were invoked.
    risk_weight(facility, pd_model=_ExplodingModel(), lgd_model=_ExplodingModel())


def test_risk_weight_raises_on_non_ttc_pd_model():
    from credit_portfolio.models import MacroConditionedPD

    facility = _wholesale_facility(ExposureClass.CORPORATE)
    with pytest.raises(ValueError, match="through-the-cycle"):
        risk_weight(facility, pd_model=MacroConditionedPD())
