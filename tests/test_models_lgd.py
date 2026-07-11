import pytest

from credit_portfolio.models import DownturnLGD, StaticLGD
from tests.conftest import make_facility


def test_static_lgd_returns_facility_lgd_unchanged():
    facility = make_facility(lgd=0.45)
    assert StaticLGD().lgd(facility) == 0.45


def test_downturn_lgd_adds_the_fixed_addon():
    facility = make_facility(lgd=0.45)
    assert DownturnLGD(addon=0.10).lgd(facility) == pytest.approx(0.55)


def test_downturn_lgd_caps_at_one():
    facility = make_facility(lgd=0.95)
    assert DownturnLGD(addon=0.20).lgd(facility) == pytest.approx(1.0)


def test_downturn_lgd_ignores_scenario():
    facility = make_facility(lgd=0.45)
    model = DownturnLGD(addon=0.10)
    assert model.lgd(facility, scenario=None) == model.lgd(facility, scenario="anything")
