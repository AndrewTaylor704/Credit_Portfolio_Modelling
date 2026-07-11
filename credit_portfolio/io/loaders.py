"""Loads a portfolio of customers and facilities from the loan-level CSV
format used by ``Dummy_loan_data.csv``.

Required columns: Customer ID, Name, SIC_Code, Country, Parent, FacID, PD,
LGD, Type, Start_date, Maturity_date, Limit, Drawn_balance, Margin, Fee,
Currency, IFRS_Stage, Turnover.

Optional columns (populate the fields added in checkpoints 2-6; omit any
and the corresponding field falls back to its domain-model default so
existing data -- e.g. ``Dummy_loan_data.csv``, which predates these columns
-- keeps loading unchanged):
  - Exposure_Class (customer-level): one of the ``ExposureClass`` enum
    names (e.g. "CORPORATE", "RETAIL_MORTGAGE", "SPECIALISED_LENDING").
    Case-insensitive. An unrecognised value is a hard error, not a silent
    fallback to Corporate -- getting this wrong misroutes IRB/standardised
    RWA to the wrong formula entirely.
  - External_Rating (customer-level): a rating bucket string (e.g. "A",
    "BBB") matching the standardised-approach tables.
  - LTV (facility-level): loan-to-value, for RETAIL_MORTGAGE standardised weights.
  - Supervisory_Slot (facility-level): one of the IRB slotting categories
    ("Strong"/"Good"/"Satisfactory"/"Weak"/"Default"), for SPECIALISED_LENDING.
  - SL_Subtype / SL_Phase (facility-level): specialised-lending standardised
    sub-type/phase (e.g. "project_finance" / "pre_operational").
  - Upfront_Fee_Rate (facility-level): one-off arrangement fee rate on the limit.

Every row is validated on load: PD/LGD must be in [0, 1], Turnover/Limit/
Drawn_balance must be non-negative, Maturity_date must be after Start_date,
and IFRS_Stage must be 1, 2, or 3. A violation raises ``ValueError``
immediately, identifying the offending Customer ID/FacID -- bad source data
should never silently reach a capital calculation.
"""

from __future__ import annotations

import pandas as pd

from credit_portfolio.domain.customer import Customer
from credit_portfolio.domain.facility import Facility
from credit_portfolio.domain.portfolio import Portfolio
from credit_portfolio.models import ExposureClass


def _optional_str(row: pd.Series, column: str) -> str | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    return str(row[column])


def _optional_float(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    return float(row[column])


def _parse_exposure_class(row: pd.Series) -> ExposureClass:
    if "Exposure_Class" not in row.index or pd.isna(row["Exposure_Class"]):
        return ExposureClass.CORPORATE
    raw = str(row["Exposure_Class"]).strip().upper()
    try:
        return ExposureClass[raw]
    except KeyError:
        raise ValueError(
            f"unknown Exposure_Class {row['Exposure_Class']!r} for customer {row['Customer ID']!r}; "
            f"expected one of {[e.name for e in ExposureClass]}"
        )


_VALID_IFRS_STAGES = (1, 2, 3)


def _validate_customer_row(row: pd.Series) -> None:
    customer_id = row["Customer ID"]
    pd_value = row["PD"]
    if pd.isna(pd_value) or not (0.0 <= pd_value <= 1.0):
        raise ValueError(f"customer {customer_id!r} has an invalid PD {pd_value!r}; expected a value in [0, 1]")
    if pd.isna(row["Turnover"]) or row["Turnover"] < 0:
        raise ValueError(f"customer {customer_id!r} has an invalid Turnover {row['Turnover']!r}; expected a non-negative value")


def _validate_facility_row(row: pd.Series) -> None:
    facid = row["FacID"]
    if pd.isna(row["LGD"]) or not (0.0 <= row["LGD"] <= 1.0):
        raise ValueError(f"facility {facid!r} has an invalid LGD {row['LGD']!r}; expected a value in [0, 1]")
    if pd.isna(row["Limit"]) or row["Limit"] < 0:
        raise ValueError(f"facility {facid!r} has an invalid Limit {row['Limit']!r}; expected a non-negative value")
    if pd.isna(row["Drawn_balance"]) or row["Drawn_balance"] < 0:
        raise ValueError(f"facility {facid!r} has an invalid Drawn_balance {row['Drawn_balance']!r}; expected a non-negative value")
    if row["Maturity_date"] <= row["Start_date"]:
        raise ValueError(
            f"facility {facid!r} has Maturity_date ({row['Maturity_date']}) at or before "
            f"Start_date ({row['Start_date']})"
        )
    if row["IFRS_Stage"] not in _VALID_IFRS_STAGES:
        raise ValueError(f"facility {facid!r} has an invalid IFRS_Stage {row['IFRS_Stage']!r}; expected one of {_VALID_IFRS_STAGES}")


def load_portfolio(csv_path: str, portfolio_id: str, portfolio_name: str) -> Portfolio:
    data = pd.read_csv(csv_path)
    data["Start_date"] = pd.to_datetime(data["Start_date"])
    data["Maturity_date"] = pd.to_datetime(data["Maturity_date"])
    data["Limit"] = data["Limit"].astype(float)
    data["Turnover"] = data["Turnover"].astype(float)
    data["Drawn_balance"] = data["Drawn_balance"].astype(float)

    optional_customer_columns = [c for c in ("Exposure_Class", "External_Rating") if c in data.columns]
    customer_data = data[
        ["Customer ID", "Name", "SIC_Code", "Country", "Parent", "PD", "Turnover"] + optional_customer_columns
    ].drop_duplicates(subset="Customer ID")

    customers_by_id: dict[str, Customer] = {}
    for _, row in customer_data.iterrows():
        _validate_customer_row(row)
        customer = Customer(
            customerid=row["Customer ID"],
            probdef=row["PD"],
            sic_code=row["SIC_Code"],
            country=row["Country"],
            name=row["Name"],
            turnover=row["Turnover"],
            parent=row["Parent"],
            exposure_class=_parse_exposure_class(row),
            external_rating=_optional_str(row, "External_Rating"),
        )
        customers_by_id[customer.customerid] = customer

    for _, row in data.iterrows():
        _validate_facility_row(row)
        customer = customers_by_id[row["Customer ID"]]
        facility = Facility(
            facid=row["FacID"],
            lgd=row["LGD"],
            type=row["Type"],
            start_date=row["Start_date"],
            maturity_date=row["Maturity_date"],
            limit=row["Limit"],
            drawn_balance=row["Drawn_balance"],
            margin=row["Margin"],
            fee=row["Fee"],
            currency=row["Currency"],
            ifrs_stage=row["IFRS_Stage"],
            customerid=row["Customer ID"],
            customer=customer,
            ltv=_optional_float(row, "LTV"),
            supervisory_slot=_optional_str(row, "Supervisory_Slot"),
            specialised_lending_subtype=_optional_str(row, "SL_Subtype"),
            specialised_lending_phase=_optional_str(row, "SL_Phase"),
            upfront_fee_rate=_optional_float(row, "Upfront_Fee_Rate"),
        )
        customer.add_facility(facility)

    portfolio = Portfolio(portfolio_id, portfolio_name)
    for customer in customers_by_id.values():
        portfolio.add_customer(customer)

    return portfolio
