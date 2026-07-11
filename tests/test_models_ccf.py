import pytest

from credit_portfolio.models import RegulatoryCCF, StaticCCF
from tests.conftest import make_facility


def test_static_ccf_returns_facility_limit_unchanged():
    facility = make_facility(limit=1_000_000.0, drawn_balance=400_000.0)
    assert StaticCCF().exposure(facility) == 1_000_000.0


def test_regulatory_ccf_applies_the_mapped_ccf_to_the_undrawn_amount():
    facility = make_facility(type="Revolver", limit=1_000_000.0, drawn_balance=400_000.0)
    # Revolver -> ccf 0.5; exposure = 400k + 0.5 * (1,000k - 400k) = 700k
    assert RegulatoryCCF().exposure(facility) == pytest.approx(700_000.0)


def test_regulatory_ccf_falls_back_to_ccf_one_for_an_unmapped_type():
    facility = make_facility(type="SomeUnknownProduct", limit=1_000_000.0, drawn_balance=400_000.0)
    assert RegulatoryCCF().exposure(facility) == pytest.approx(1_000_000.0)


def test_regulatory_ccf_clamps_when_overdrawn():
    facility = make_facility(type="Revolver", limit=500_000.0, drawn_balance=600_000.0)
    # Overdrawn: undrawn would be negative, must clamp to 0 so exposure never drops below drawn balance.
    assert RegulatoryCCF().exposure(facility) == pytest.approx(600_000.0)
