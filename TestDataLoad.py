import CreditObjects
from csv import DictReader

with open('Dummy_loan_data.csv') as f:
    facilities = []
    customers = []
    reader = DictReader(f)
    for row in reader:
        facilities.append(Facility())
