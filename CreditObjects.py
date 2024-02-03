import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import torch

class Facility():
    def __init__(self, facid, lgd, type, start_date, maturity_date, limit, drawn_balance, margin, fee, 
                 currency, ifrs_stage, customerid, customer):
        self.facid = facid
        self.lgd = lgd
        self.type = type
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.limit = limit
        self.drawn_balance = drawn_balance
        self.margin = margin
        self.fee = fee
        self.currency = currency
        self.ifrs_stage = ifrs_stage
        self.customerid = customerid
        self.customer = customer

    def ead(self):
        if self.type == 'Loan':
            ead = self.limit
        else:
            ead = self.limit
        return ead

    def ecl(self):
        ecl = self.lgd * self.ead() * self.customer.probdef
        ecl = ecl
        return ecl
    
    def risk_weight(self):
        #insert risk weight calc here
        risk_weight = 0.05
        return risk_weight
    
    def rwa(self):
        rwa = self.ead() * self.risk_weight()
        return rwa 
    
class Customer():
    def __init__(self, customerid, probdef, sic_code, country, name, parent = None):
        self.customerid = customerid
        self.probdef = probdef
        self.sic_code = sic_code
        self.country = country
        self.name = name
        self.parent = parent
        self.facility_list = []

    def add_facility(self, facid):
        self.facility_list.append(facid)
        
    def ecl(self):
        self.ecl = 0
        for facility in self.facility_list:
            self.ecl += facility.ecl()
        return self.ecl
    
def return_customers(facid):
    tempcust = []
    for i in range(len(customers)):
        if customers[i].customerid == facid:
            return customers[i]

data = pd.read_csv('Dummy_loan_data.csv')
data['Start_date'] = pd.to_datetime(data['Start_date'])
data['Maturity_date'] = pd.to_datetime(data['Maturity_date'])
data['Limit'] = data['Limit'].astype('float')
data['Drawn_balance'] = data['Drawn_balance'].astype('float')

customerdata = data[['Customer ID', 'Name', 'SIC_Code', 'Country', 'Parent', 'PD']]
customerdata = customerdata.drop_duplicates(subset='Customer ID')

facilities = []
customers = []

for i in customerdata.index:
    customer = Customer(customerdata['Customer ID'][i], customerdata['PD'][i], customerdata['SIC_Code'][i],
                        customerdata['Country'][i], customerdata['Name'][i],customerdata['Parent'][i])
    customers.append(customer)

for i in data.index:
    facility = Facility(data['FacID'][i],data['LGD'][i], data['Type'][i], 
                        data['Start_date'][i], data['Maturity_date'][i], 
                        data['Limit'][i], data['Drawn_balance'][i], 
                        data['Margin'][i], data['Fee'][i], data['Currency'][i], 
                        data['IFRS_Stage'][i], data['Customer ID'][i], 
                        return_customers(data['Customer ID'][i]))
    facilities.append(facility)
    for j in range(len(customers)):
        if data['Customer ID'][i] == customers[j].customerid:
            customers[j].add_facility(data['FacID'][i]) 

print(facilities[1].lgd)
print(facilities[1].customer)
print(facilities[1].limit)
print(facilities[1].ead())
print(facilities[1].customer.probdef)
print(facilities[1].ecl())
print(facilities[1].rwa())
