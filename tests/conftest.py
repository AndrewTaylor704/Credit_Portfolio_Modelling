from datetime import datetime

from credit_portfolio.amortisation import AmortisationSchedule, Frequency
from credit_portfolio.domain import Customer, Facility
from credit_portfolio.models import ExposureClass


def make_customer(**overrides) -> Customer:
    defaults = dict(
        customerid="C1",
        probdef=0.01,
        sic_code="1234",
        country="GB",
        name="Test Customer",
        turnover=30_000_000,
        parent=None,
        exposure_class=ExposureClass.CORPORATE,
        external_rating=None,
    )
    defaults.update(overrides)
    return Customer(**defaults)


def make_facility(customer: Customer | None = None, **overrides) -> Facility:
    customer = customer or make_customer()
    defaults = dict(
        facid="F1",
        lgd=0.45,
        type="Loan",
        start_date=datetime(2020, 1, 1),
        maturity_date=datetime(2027, 1, 1),
        limit=1_000_000.0,
        drawn_balance=1_000_000.0,
        margin=0.03,
        fee=0.005,
        currency="GBP",
        ifrs_stage=1,
        customerid=customer.customerid,
        customer=customer,
        amort_schedule=AmortisationSchedule.straight_line(Frequency.QUARTERLY),
    )
    defaults.update(overrides)
    facility = Facility(**defaults)
    customer.add_facility(facility)
    return facility
