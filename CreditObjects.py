import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class Facility():
    def __init__(self, facid, lgd, type, start_date, maturity_date, limit, drawn_balance, margin, fee, currency, ifrs_stage, customerid):
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

    def ead(self):
        if self.type == 'loan':
            self.ead = self.limit
        else:
            self.ead = self.limit

    def ecl(self):
        self.ecl = self.lgd * self.ead * self.customerid.probdef
        return self.ecl
    
    def risk_weight(self):
        #insert risk weight calc here
        return self.risk_weight
    
    def rwa(self):
        self.rwa = self.ead*self.risk_weight
        return self.rwa 
    
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