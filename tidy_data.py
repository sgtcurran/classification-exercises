#%%
import pandas as pd
import numpy as np
import acquire 
from env import host, user, password, get_db_url
#%%
# Exercie 1 Attendance Data
# Read the data from the attendance table and calculate an 
# attendance percentage for each student. One half day is 
# worth 50% of a full day, and 10 tardies is equal to one absence.

attend_df = acquire.get_attendance_data()
attend_df
#%%
attend_df.columns
#%%
attend_df.rename(columns={'Unnamed: 0': 'name'}, inplace=True)
attend_df
#%%
attend_melt = attend_df.melt(id_vars='name', var_name='date', value_name='attendance').drop(columns='date')
attend_melt
#%%
attend_melt['attendance'] = attend_melt['attendance'].replace('P', 1)
attend_melt['attendance'] = attend_melt['attendance'].replace('A', 0)
attend_melt['attendance'] = attend_melt['attendance'].replace('T', 0.9)
attend_melt['attendance'] = attend_melt['attendance'].replace('H', 0.5)
attend_melt.groupby('name')['attendance'].mean()


#%%



# %%


