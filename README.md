# Credit Portfolio Modelling

A credit portfolio analytics toolkit: risk parameters, IRB and standardised
RWA, expected credit loss (including IFRS 9 staging), Monte Carlo economic
capital, facility/customer cashflow projection, RAROC, and structured-finance
(securitisation) tranching and cash waterfalls — plus a Streamlit UI over all
of it.

## Layout

```
credit_portfolio/
    domain/            Facility/Customer/Portfolio data holders (no cached valuation)
                        + a currency-consistency guard every aggregation calls first
    amortisation/        Flexible, rule-based facility amortisation schedule engine
                        (bullet, straight-line, or custom hybrid schedules)
    models/                Pluggable risk-parameter strategies: PD, LGD, CCF, funding
                        cost, operating cost, exposure class, scenario (macro Z-scores)
    valuation/               EAD / ECL (incl. IFRS 9 stage-aware ifrs9_ecl) / RWA, as
                        explicit functions of an as-of date and scenario:
                          - IRB: wholesale, retail, and specialised-lending formulas
                          - Standardised: BCBS baseline, plus EU (CRR3) and UK
                            (PRA Basel 3.1) unrated-corporate/SME divergences
    risk/                      Monte Carlo simulation:
                          - single-period loss distribution + tranche loss allocation
                          - multi-period default-timing simulation for cashflow paths
    cashflow/                    Facility/customer/portfolio income + principal
                        cashflow projection, including default/recovery scenarios
                        (cross-default at the obligor level)
    raroc/                          Risk-adjusted return on capital: regulatory or
                        (standalone, non-diversified) economic capital basis
    securitisation/                    Tranche loss allocation and a pro-rata
                        pass-through cash waterfall
    calibration/                          A live-data-informed macro/equity Z-score
                        demonstration dataset for MacroConditionedPD
    io/                                     CSV loading, with input validation
scripts/
    run_dummy_portfolio.py                   Demo driver against Dummy_loan_data.csv
streamlit_app.py, streamlit_common.py, pages/    Multi-page Streamlit UI
tests/                                              21 test modules, 123 tests
```

## Setup

```
pip install -r requirements.txt
```

## Running the demo script

```
python scripts/run_dummy_portfolio.py
```

Loads `Dummy_loan_data.csv`, prints portfolio RWA/UL, runs the Monte Carlo
loss simulation, and plots the loss distribution.

## Running the UI

```
streamlit run streamlit_app.py
```

Load a portfolio on the Home page (the dummy CSV, or upload your own with the
same schema — see below), then use the sidebar to move between RWA & ECL,
Loss Distribution, RAROC, Cashflow, and Tranching & Waterfall.

## Running the tests

```
pytest tests/
```

## Data schema

Required CSV columns: `Customer ID, Name, SIC_Code, Country, Parent, FacID,
PD, LGD, Type, Start_date, Maturity_date, Limit, Drawn_balance, Margin, Fee,
Currency, IFRS_Stage, Turnover`.

Optional columns (each falls back to a sensible default if omitted, so
existing data keeps loading unchanged):

| Column | Level | Populates |
|---|---|---|
| `Exposure_Class` | customer | `ExposureClass` enum name (e.g. `CORPORATE`, `RETAIL_MORTGAGE`, `SPECIALISED_LENDING`); case-insensitive, unknown values are a hard error |
| `External_Rating` | customer | rating bucket for the standardised-approach tables (e.g. `A`, `BBB`) |
| `LTV` | facility | loan-to-value, for `RETAIL_MORTGAGE` standardised weights |
| `Supervisory_Slot` | facility | IRB slotting category, for `SPECIALISED_LENDING` |
| `SL_Subtype` / `SL_Phase` | facility | specialised-lending standardised sub-type/phase |
| `Upfront_Fee_Rate` | facility | one-off arrangement fee rate, accrued over the facility's life |

Every row is validated on load: PD/LGD in [0, 1], non-negative
Turnover/Limit/Drawn_balance, `Maturity_date` after `Start_date`, and
`IFRS_Stage` in {1, 2, 3}. A violation raises immediately, naming the
offending Customer ID/FacID.

Facilities in more than one `Currency` cannot be aggregated together — there
is no FX conversion in this codebase, so mixed-currency aggregation raises
rather than silently blending amounts.

## Known limitations

This toolkit's formulas are real and tested, but several inputs are
currently placeholders rather than calibrated models, and a few structural
simplifications are documented in the code where they're made:

- **No real PD, LGD, or CCF model** — `StaticPD`/`StaticLGD`/`StaticCCF`
  pass through whatever's in the input data; `MacroConditionedPD` is a
  formula, not an estimator. Plugging in a real model (scorecard,
  structural, ML) is a matter of implementing the corresponding protocol in
  `credit_portfolio/models/`.
- **Funding cost and operating cost are flat, user-supplied rates**
  (`FlatFundingRate`, `FlatOperatingCostRate`), not real curves or
  cost-to-serve data.
- **Monte Carlo correlation is a manual input**, not calibrated from
  historical default co-movement.
- **The macro Z-score dataset is a small, cited demonstration** (three
  indicators; see `credit_portfolio/calibration/live_data.py`), not full
  coverage of a real portfolio's sector/region mix.
- **Economic capital is standalone (non-diversified)**, not a Monte
  Carlo-allocated share of true portfolio-diversified capital.
- **The cash waterfall is pro-rata pass-through**, not a sequential-pay
  structure with independent per-tranche coupons.
- **EU/UK standardised RWA only diverges from the BCBS baseline for
  Corporate/SME unrated exposures** — other exposure classes use the BCBS
  tables for all jurisdictions.
- No model governance/audit trail, no persistence layer, and no deployment
  hardening (auth, multi-user support) — this runs as a single-user local
  tool today.

Each of the above is a deliberate, documented choice (see the relevant
module's docstring), not an oversight — check the module you're relying on
before using its output for a real decision.
