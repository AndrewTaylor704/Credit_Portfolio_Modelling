# Credit_Portfolio_Modelling

Various credit analytics calculations.

## Layout

```
credit_portfolio/
    domain/          # Facility/Customer/Portfolio data holders (no cached valuation)
    amortisation/     # Flexible facility amortisation schedule engine
    valuation/         # EAD/ECL/RWA (IRB) as explicit functions of an as-of date
    risk/                # Single-factor Gaussian copula Monte Carlo loss simulation
    io/                   # CSV loading
    cashflow/, raroc/, securitisation/   # scaffolds for future work, not implemented yet
scripts/
    run_dummy_portfolio.py   # demo driver against Dummy_loan_data.csv
tests/
```

## Running the demo

```
python scripts/run_dummy_portfolio.py
```

## Running the tests

```
pytest tests/
```
