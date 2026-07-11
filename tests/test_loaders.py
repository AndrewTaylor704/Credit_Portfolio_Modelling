import pandas as pd
import pytest

from credit_portfolio.io import load_portfolio
from credit_portfolio.models import ExposureClass

REQUIRED_COLUMNS = {
    "Customer ID": "C1",
    "Name": "Test Co",
    "SIC_Code": "1234",
    "Country": "GB",
    "Parent": None,
    "FacID": "F1",
    "PD": 0.02,
    "LGD": 0.45,
    "Type": "Loan",
    "Start_date": "2020-01-01",
    "Maturity_date": "2027-01-01",
    "Limit": 1_000_000.0,
    "Drawn_balance": 1_000_000.0,
    "Margin": 0.03,
    "Fee": 0.01,
    "Currency": "GBP",
    "IFRS_Stage": 1,
    "Turnover": 30_000_000.0,
}


def _write_csv(tmp_path, rows):
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_loading_without_optional_columns_defaults_as_before(tmp_path):
    csv_path = _write_csv(tmp_path, [REQUIRED_COLUMNS])
    portfolio = load_portfolio(csv_path, "P1", "test")

    customer = portfolio.customer_list[0]
    facility = customer.facility_list[0]
    assert customer.exposure_class is ExposureClass.CORPORATE
    assert customer.external_rating is None
    assert facility.ltv is None
    assert facility.supervisory_slot is None
    assert facility.specialised_lending_subtype is None
    assert facility.specialised_lending_phase is None
    assert facility.upfront_fee_rate is None


def test_loading_with_optional_columns_populates_new_fields(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row.update({
        "Exposure_Class": "retail_mortgage",  # lower-case, should be case-insensitive
        "External_Rating": "A",
        "LTV": 0.65,
        "Upfront_Fee_Rate": 0.01,
    })
    csv_path = _write_csv(tmp_path, [row])
    portfolio = load_portfolio(csv_path, "P1", "test")

    customer = portfolio.customer_list[0]
    facility = customer.facility_list[0]
    assert customer.exposure_class is ExposureClass.RETAIL_MORTGAGE
    assert customer.external_rating == "A"
    assert facility.ltv == pytest.approx(0.65)
    assert facility.upfront_fee_rate == pytest.approx(0.01)


def test_specialised_lending_columns_populate_facility_fields(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row.update({
        "Exposure_Class": "SPECIALISED_LENDING",
        "Supervisory_Slot": "Good",
        "SL_Subtype": "project_finance",
        "SL_Phase": "pre_operational",
    })
    csv_path = _write_csv(tmp_path, [row])
    portfolio = load_portfolio(csv_path, "P1", "test")

    customer = portfolio.customer_list[0]
    facility = customer.facility_list[0]
    assert customer.exposure_class is ExposureClass.SPECIALISED_LENDING
    assert facility.supervisory_slot == "Good"
    assert facility.specialised_lending_subtype == "project_finance"
    assert facility.specialised_lending_phase == "pre_operational"


def test_unknown_exposure_class_raises_a_clear_error(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row["Exposure_Class"] = "NOT_A_REAL_CLASS"
    csv_path = _write_csv(tmp_path, [row])

    with pytest.raises(ValueError, match="unknown Exposure_Class"):
        load_portfolio(csv_path, "P1", "test")


def test_pd_out_of_range_raises(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row["PD"] = 1.5
    csv_path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match="invalid PD"):
        load_portfolio(csv_path, "P1", "test")


def test_negative_turnover_raises(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row["Turnover"] = -1.0
    csv_path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match="invalid Turnover"):
        load_portfolio(csv_path, "P1", "test")


def test_lgd_out_of_range_raises(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row["LGD"] = -0.1
    csv_path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match="invalid LGD"):
        load_portfolio(csv_path, "P1", "test")


def test_negative_limit_raises(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row["Limit"] = -1_000.0
    csv_path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match="invalid Limit"):
        load_portfolio(csv_path, "P1", "test")


def test_negative_drawn_balance_raises(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row["Drawn_balance"] = -1_000.0
    csv_path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match="invalid Drawn_balance"):
        load_portfolio(csv_path, "P1", "test")


def test_maturity_before_start_raises(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row["Maturity_date"] = "2019-01-01"  # before Start_date
    csv_path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match="Maturity_date"):
        load_portfolio(csv_path, "P1", "test")


def test_invalid_ifrs_stage_raises(tmp_path):
    row = dict(REQUIRED_COLUMNS)
    row["IFRS_Stage"] = 7
    csv_path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match="invalid IFRS_Stage"):
        load_portfolio(csv_path, "P1", "test")
