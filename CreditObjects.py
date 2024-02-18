import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy.stats import norm

class Facility():
    def __init__(self, facid, lgd, type, start_date, maturity_date, limit, drawn_balance, margin, fee, 
                 currency, ifrs_stage, customerid, customer, frequency = 'quarterly'):
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
        self.maturity = (maturity_date - dt.datetime.now()).days / 365.25
        self.effmaturity = max(min(self.maturity, 5), 1)
        self.frequency = frequency
        self.ead = self.ead()
        self.ecl = self.ecl()
        self.risk_weight = self.risk_weight()
        self.rwa = self.rwa()
        self.calc_amort_profile()

    def calc_amort_profile(self):
        #Straight line amortisation profiles only at the moment, need to expand to mortgage style and bullets
        self.amort_profile = []
        if self.frequency == 'quarterly':
            payments_per_year = 4.0
        elif self.frequency == 'monthly':
            payments_per_year = 12.0
        elif self.frequency == 'annual':
            payments_per_year = 1.0
        else:
            raise ValueError('Incorrect payment frequency type')
        
        remaining_payments = np.ceil(self.maturity * payments_per_year)
        payment_amount = self.drawn_balance / remaining_payments

        for i in range(int(remaining_payments + 1),0 , -1):
            j = int(remaining_payments) - i + 1
            self.amort_profile.insert(0, (self.maturity_date - (j * dt.timedelta(days = 365.25/payments_per_year)), j * payment_amount))
        return self.amort_profile
    
    def balance_on_date(self, future_date):
        self.calc_amort_profile()
        future_date = pd.to_datetime(future_date)
        dates_list = [list(t) for t in zip(*self.amort_profile)][0]
        idx = 0
        if future_date > self.maturity_date:
            idx = len(self.amort_profile)
        elif future_date <= dt.datetime.now():
            idx = 1
        else:
            while future_date > dates_list[idx]:
                idx += 1
        return self.amort_profile[idx-1][1]

    def ead(self):
        if self.type == 'Loan':
            ead = self.limit
        else:
            ead = self.limit
        return ead

    def ecl(self):
        ecl = self.lgd * self.ead * self.customer.probdef
        return ecl
    
    def calc_risk_weight(self, pd, lgd, maturity, turnover):
        # Note this just does the Corporate RWA calc today - needs expanding to cover Retail
        effmat = max(min(self.maturity, 5), 1)
        R = (0.12 * (1 - np.exp(-50*pd))/(1-np.exp(-50))) + (0.24 * (1-(1 - np.exp(-50*pd))/(1-np.exp(-50))))
        if turnover <= 50:
            R += -0.04 * (1 - ((max(turnover, 5) - 5) / 45))
        b = (0.11852 - (0.05478 * np.log(pd)))**2
        matadj = (1 + ((effmat - 2.5) * b))/(1 - (1.5 * b))
        K = (lgd * norm.cdf(norm.ppf(pd)/np.sqrt(1-R)+np.sqrt(R/(1-R))*norm.ppf(0.999)) 
             - (lgd * pd)) * matadj
        calc_risk_weight = K * 12.5
        return calc_risk_weight

    def risk_weight(self):
        risk_weight = self.calc_risk_weight(self.customer.probdef, self.lgd, self.maturity, self.customer.turnover)
        return risk_weight
    
    def rwa(self):
        rwa = self.ead * self.risk_weight
        return rwa 
    
class Customer():
    def __init__(self, customerid, probdef, sic_code, country, name, turnover, parent = None):
        self.customerid = customerid
        self.probdef = probdef
        self.sic_code = sic_code
        self.country = country
        self.name = name
        self.parent = parent
        self.turnover = turnover / 1000000
        self.facility_list = []

    def add_facility(self, facility):
        self.facility_list.append(facility)
        
    def ecl(self):
        ecl = 0
        for facility in self.facility_list:
            ecl += facility.ecl
        return ecl
    
    def rwa(self):
        rwa = 0
        for facility in self.facility_list:
            rwa += facility.rwa
        return rwa

    def ead(self):
        ead = 0
        for facility in self.facility_list:
            ead += facility.ead
        return ead
    
    def risk_weight(self):
        risk_weight = self.rwa() / self.ead()
        return risk_weight

    def balance_on_date(self, future_date):
        balance = 0
        for facility in self.facility_list:
            balance += facility.balance_on_date(future_date)
        return balance

    def calc_losses(self, future_date):
        losses = 0
        for facility in self.facility_list:
            losses += facility.balance_on_date(future_date) * facility.lgd
        return losses

class Portfolio():
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.customer_list = []
        self.num_customers = 0

    def add_customer(self, customer):
        self.customer_list.append(customer)
        self.num_customers += 1

    def ecl(self):
        ecl = 0
        for customer in self.customer_list:
            ecl += customer.ecl()
        return ecl
    
    def rwa(self):
        rwa = 0
        for customer in self.customer_list:
            rwa += customer.rwa()
        return rwa

    def ead(self):
        ead = 0
        for customer in self.customer_list:
            ead += customer.ead
        return ead
    
    def risk_weight(self):
        risk_weight = self.rwa() / self.ead()
        return risk_weight

    def balance_on_date(self, future_date):
        balance = 0
        for customer in self.customer_list:
            balance += customer.balance_on_date(future_date)
        return balance

    def calc_losses(self, future_date):
        losses = 0
        for customer in self.customer_list:
            losses += customer.calc_losses(future_date)
        return losses

    def simulate_loss_distribution(self, correlation, num_sims, num_bins, horizon = 1, random_seed = 1234):
        threshold_value = np.zeros(self.num_customers)
        counter = 0
        rng = np.random.default_rng(seed = random_seed)
        potential_loss = np.zeros(self.num_customers)
        loss_dist = np.zeros((num_bins,2))
        total_notional = 0
        for customer in self.customer_list:
            threshold_value[counter] = norm.ppf(customer.probdef)
            potential_loss[counter] = customer.calc_losses(dt.timedelta(days = horizon * 365.25) + dt.datetime.now())
            total_notional += potential_loss[counter]
            counter += 1

        binsize = total_notional / num_bins

        for i in range(0, num_sims):
            if i // np.ceil(num_sims/10) == i / np.ceil(num_sims/10):
                print(f"Simulation # {i}")
            sim_loss = 0
            bin_loss = 0
            default_marker = np.zeros(self.num_customers)
            syst_rand = rng.standard_normal()
            idio_rand = rng.standard_normal(self.num_customers)
            cust_asset_value = np.sqrt(correlation) * syst_rand + np.sqrt(1-correlation) * idio_rand

            for k in range(0, self.num_customers):
                if cust_asset_value[k] < threshold_value[k]:
                    default_marker[k] = True
                else:
                    default_marker[k] = False

            sim_loss = np.sum(default_marker * potential_loss)
            bin_loss = min(sim_loss / total_notional * num_bins, num_bins-1)
            
            j = 0
            while j <= bin_loss:
                loss_dist[j,0] += 1
                j += 1
        loss_dist = loss_dist / num_sims
        for i in range(0, num_bins):
            loss_dist[i, 1] = binsize * i

        self.loss_dist = loss_dist

    def plot_loss_dist(self):
        plt.plot(self.loss_dist[:,1], self.loss_dist[:,0])
        plt.show()

    def loss_dist_quantile(self, confidence):
        for i in range(0, self.loss_dist.shape[0]-1):
            if self.loss_dist[i,0] > 1 - confidence > self.loss_dist[i+1, 0]:
                return self.loss_dist[i,1]
        return False

def return_customers(facid):
    tempcust = []
    for i in range(len(customers)):
        if customers[i].customerid == facid:
            return customers[i]

data = pd.read_csv('Dummy_loan_data.csv')
data['Start_date'] = pd.to_datetime(data['Start_date'])
data['Maturity_date'] = pd.to_datetime(data['Maturity_date'])
data['Limit'] = data['Limit'].astype('float')
data['Turnover'] = data['Turnover'].astype('float')
data['Drawn_balance'] = data['Drawn_balance'].astype('float')

customerdata = data[['Customer ID', 'Name', 'SIC_Code', 'Country', 'Parent', 'PD', 'Turnover']]
customerdata = customerdata.drop_duplicates(subset='Customer ID')

facilities = []
customers = []

for i in customerdata.index:
    customer = Customer(customerdata['Customer ID'][i], customerdata['PD'][i], customerdata['SIC_Code'][i],
                        customerdata['Country'][i], customerdata['Name'][i],customerdata['Turnover'][i],
                        customerdata['Parent'][i])
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

portfolio = Portfolio('10001', 'Dummy portfolio')

for i, x in enumerate(customers):
    portfolio.add_customer(x)

# print(facilities[1].lgd)
# print(facilities[1].customer)
# print(facilities[1].limit)
# print(facilities[1].ead)
# print(facilities[1].customer.probdef)
# print(facilities[1].ecl)
# print(facilities[1].rwa)
# print(facilities[1].maturity)
# print(facilities[1].effmaturity)
# print(facilities[0].risk_weight)
# print(facilities[6].rwa)
# print(portfolio.ecl())
# print(customers[0].ead())
# print(facilities[6].calc_amort_profile())            
# print(f'rwa = {portfolio.rwa():.4f}')
# print(portfolio.customer_list)

print(portfolio.balance_on_date(dt.timedelta(days = 365.25) + dt.datetime.now()))
print(portfolio.calc_losses(dt.timedelta(days = 365.25) + dt.datetime.now()))
portfolio.simulate_loss_distribution(0.15, 10000, 1000)
portfolio.plot_loss_dist()
print(portfolio.loss_dist_quantile(0.999))
print(portfolio.loss_dist)
# print(portfolio.loss_dist_quantile(0.99))
# print(portfolio.loss_dist_quantile(0.95))