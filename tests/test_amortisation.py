from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from credit_portfolio.amortisation import (
    AmortMethod,
    AmortRule,
    AmortisationSchedule,
    Frequency,
    RelativeAnchor,
    resolve,
)

START = date(2020, 1, 1)
MATURITY = date(2025, 1, 1)  # exactly 5 years
BALANCE = 1_000_000.0


def test_bullet_schedule_pays_full_balance_at_maturity():
    schedule = AmortisationSchedule.bullet()
    result = resolve(schedule, START, MATURITY, BALANCE)

    assert len(result.payments) == 1
    payment = result.payments[0]
    assert payment.date == MATURITY
    assert payment.opening_balance == pytest.approx(BALANCE)
    assert payment.principal_payment == pytest.approx(BALANCE)
    assert payment.closing_balance == pytest.approx(0.0)


def test_straight_line_quarterly_amortises_evenly_to_zero():
    schedule = AmortisationSchedule.straight_line(Frequency.QUARTERLY)
    result = resolve(schedule, START, MATURITY, BALANCE)

    assert len(result.payments) == 20  # 5 years * 4 quarters
    expected_payment = BALANCE / 20
    for payment in result.payments:
        assert payment.principal_payment == pytest.approx(expected_payment)
    assert result.payments[-1].date == MATURITY
    assert result.payments[-1].closing_balance == pytest.approx(0.0)


def test_hybrid_schedule_no_amort_then_percent_then_bullet():
    # No scheduled amort for years 0-3, then 1.5%/yr of original balance
    # (annual) for years 3-5, final bullet for the remainder.
    rules = [
        AmortRule(
            method=AmortMethod.BULLET,
            window_start=RelativeAnchor.from_start(),
            window_end=RelativeAnchor.from_maturity(years=-2),
        ),
        AmortRule(
            method=AmortMethod.PERCENT_OF_ORIGINAL,
            window_start=RelativeAnchor.from_maturity(years=-2),
            window_end=RelativeAnchor.from_maturity(),
            frequency=Frequency.ANNUAL,
            rate=0.015,
        ),
    ]
    schedule = AmortisationSchedule.from_rules(rules)
    result = resolve(schedule, START, MATURITY, BALANCE)

    dates = [p.date for p in result.payments]
    assert dates == [date(2024, 1, 1), MATURITY]

    first_payment = result.payments[0]
    assert first_payment.principal_payment == pytest.approx(BALANCE * 0.015)

    final_payment = result.payments[-1]
    assert final_payment.closing_balance == pytest.approx(0.0)
    assert final_payment.principal_payment == pytest.approx(BALANCE * (1 - 0.015))


def test_segments_constructor_matches_from_rules_for_hybrid_case():
    hand_built = AmortisationSchedule.from_rules([
        AmortRule(
            method=AmortMethod.BULLET,
            window_start=RelativeAnchor.from_start(),
            window_end=RelativeAnchor.from_maturity(years=-2),
        ),
        AmortRule(
            method=AmortMethod.PERCENT_OF_ORIGINAL,
            window_start=RelativeAnchor.from_maturity(years=-2),
            window_end=RelativeAnchor.from_maturity(),
            frequency=Frequency.ANNUAL,
            rate=0.015,
        ),
    ])
    chained = AmortisationSchedule.segments([
        (relativedelta(years=3), AmortRule(method=AmortMethod.BULLET)),
        (None, AmortRule(method=AmortMethod.PERCENT_OF_ORIGINAL, frequency=Frequency.ANNUAL, rate=0.015)),
    ])

    assert resolve(hand_built, START, MATURITY, BALANCE) == resolve(chained, START, MATURITY, BALANCE)


def test_validate_rejects_gap_between_rules():
    rules = [
        AmortRule(
            method=AmortMethod.BULLET,
            window_start=RelativeAnchor.from_start(),
            window_end=RelativeAnchor.from_maturity(years=-3),  # leaves years -3..-2 uncovered
        ),
        AmortRule(
            method=AmortMethod.PERCENT_OF_ORIGINAL,
            window_start=RelativeAnchor.from_maturity(years=-2),
            window_end=RelativeAnchor.from_maturity(),
            frequency=Frequency.ANNUAL,
            rate=0.015,
        ),
    ]
    schedule = AmortisationSchedule.from_rules(rules)
    with pytest.raises(ValueError, match="gap"):
        schedule.validate(START, MATURITY)


def test_validate_rejects_overlap_between_rules():
    rules = [
        AmortRule(
            method=AmortMethod.BULLET,
            window_start=RelativeAnchor.from_start(),
            window_end=RelativeAnchor.from_maturity(years=-1),  # overlaps the next rule's window
        ),
        AmortRule(
            method=AmortMethod.PERCENT_OF_ORIGINAL,
            window_start=RelativeAnchor.from_maturity(years=-2),
            window_end=RelativeAnchor.from_maturity(),
            frequency=Frequency.ANNUAL,
            rate=0.015,
        ),
    ]
    schedule = AmortisationSchedule.from_rules(rules)
    with pytest.raises(ValueError, match="overlap"):
        schedule.validate(START, MATURITY)


def test_over_amortising_rules_warn_and_clamp():
    # Two PERCENT_OF_ORIGINAL rules at 60% each, annual, over the last 2 years -
    # 120% of original balance scheduled against a facility that only owes 100%.
    rules = [
        AmortRule(
            method=AmortMethod.BULLET,
            window_start=RelativeAnchor.from_start(),
            window_end=RelativeAnchor.from_maturity(years=-2),
        ),
        AmortRule(
            method=AmortMethod.PERCENT_OF_ORIGINAL,
            window_start=RelativeAnchor.from_maturity(years=-2),
            window_end=RelativeAnchor.from_maturity(),
            frequency=Frequency.ANNUAL,
            rate=0.6,
        ),
    ]
    schedule = AmortisationSchedule.from_rules(rules)

    with pytest.warns(RuntimeWarning, match="over-amortise"):
        result = resolve(schedule, START, MATURITY, BALANCE)

    # Balance never goes negative despite the 120% of scheduled principal.
    assert all(p.closing_balance >= 0 for p in result.payments)
    assert result.payments[-1].closing_balance == pytest.approx(0.0)


def test_balance_on_date_at_seam_mid_period_and_boundaries():
    schedule = AmortisationSchedule.straight_line(Frequency.ANNUAL)
    result = resolve(schedule, START, MATURITY, BALANCE)

    # Before the first payment: full balance.
    assert result.balance_on_date(date(2020, 6, 1)) == pytest.approx(BALANCE)
    # Exactly on a payment date (a "seam"): balance already reflects that payment
    # (2 of 5 annual payments made, 3/5 of original balance remaining).
    assert result.balance_on_date(date(2022, 1, 1)) == pytest.approx(BALANCE * 3 / 5)
    # Mid-period, between payments: balance reflects the most recent payment.
    assert result.balance_on_date(date(2022, 6, 1)) == pytest.approx(BALANCE * 3 / 5)
    # After maturity: fully repaid.
    assert result.balance_on_date(date(2026, 1, 1)) == pytest.approx(0.0)
