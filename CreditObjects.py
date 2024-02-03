import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from  datetime import datetime
from scipy.stats import norm
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
        self.maturity = (maturity_date - datetime.now()).days / 365.25
        self.effmaturity = max(min(self.maturity, 5), 1)

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
        R = (0.12 * (1 - np.exp(-50*self.customer.probdef))/(1-np.exp(-50))) + (0.24 * (1-(1 - np.exp(-50*self.customer.probdef))/(1-np.exp(-50))))
        b = (0.11852 - (0.05478 * np.log(self.customer.probdef)))**2
        matadj = (1 + ((self.effmaturity  - 2.5) * b))/(1 - (1.5 * b))
        temp = norm.ppf(self.customer.probdef)/np.sqrt(1-R)+np.sqrt(R/(1-R))*norm.ppf(0.999) 
        K = (self.lgd * norm.cdf(norm.ppf(self.customer.probdef)/np.sqrt(1-R)+np.sqrt(R/(1-R))*norm.ppf(0.999)) 
             - (self.lgd * self.customer.probdef)) * matadj
        risk_weight = K * 12.5
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

    def add_facility(self, facility):
        self.facility_list.append(facility)
        
    def ecl(self):
        ecl = 0
        for facility in self.facility_list:
            ecl += facility.ecl()
        return ecl
    
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
            customers[j].add_facility(facility) 

# print(facilities[1].lgd)
# print(facilities[1].customer)
# print(facilities[1].limit)
# print(facilities[1].ead())
# print(facilities[1].customer.probdef)
# print(facilities[1].ecl())
# print(facilities[1].rwa())
# print(facilities[1].maturity)
# print(facilities[1].effmaturity)
# print(facilities[0].risk_weight())
# print(facilities[1].rwa())

# print(customers[1].ecl())