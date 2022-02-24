#%%
from webbrowser import get
from env import host, user, password
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
from tabulate import tabulate

#%%
def get_titanic_data():
    filename = 'titanic_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'titanic_db'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'

    query = "SELECT* FROM passengers"


    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
# %%
get_titanic_data()
# %%
# The returned data frame should include the actual name 
# of the species in addition to the species_ids.
#%%
def get_iris_data():
    filename = 'iris_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'iris_db'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'

    query = '''
SELECT measurement_id, species_id, species_name
FROM species
JOIN measurements USING (species_id)



'''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
# %%
get_iris_data()
# %%
# join all 4 tables together, so that the resulting dataframe 
# contains all the contract, payment, and internet service options.
#%%
def get_telco_data():
    filename = 'telco_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'telco_churn'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'

    query = '''
SELECT*
FROM customers
JOIN customer_contracts USING (customer_id)
JOIN customer_payments USING (Customer_id)
JOIN internet_service_types

'''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
#%%
get_telco_data()
#%%
print(tabulate(get_telco_data(), headers = 'keys', tablefmt = 'psql'))
# %%
