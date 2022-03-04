#%%
from cProfile import label
import pandas as pd
import numpy as np
import acquire 
from env import host, user, password, get_db_url
import seaborn as sns

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
# Exercise 2 Coffee Levels
# Read the coffee_levels table
# Transform the data so that each carafe is in it's own column.
# Is this the best shape for the data?
df = acquire.get_coffee_data()
df.info()
df.shape
df.head()
#%%
df['coffee_carafe_x'] = df['coffee_carafe'].where(df['coffee_carafe'] == 'x').fillna('')
df['coffee_carafe_y'] = df['coffee_carafe'].where(df['coffee_carafe'] == 'y').fillna('')
df['coffee_carafe_z'] = df['coffee_carafe'].where(df['coffee_carafe'] == 'z').fillna('')
df_drop = df.drop(columns='coffee_carafe')
df_drop.info()
#%%
cafe = 'coffee_carafe_x','coffee_carafe_y','coffee_carafe_z'
coffee = df_drop.pivot(index=['hour'],columns=cafe, values='coffee_amount')
coffee.reset_index()
#%%
coffee.columns.names = [None, None, None]
coffee
#%%
# exercise 2 c. Is the data tidy I guess. But the information is there if want to know the detaila
# by just eyeballing it.
# %%
# exercise 3
cake_df = acquire.get_cake_data()
cake_df.info()
#%%
cake_df.head()
# %%
cake_split = cake_df
cake_split
# %%
cake_split['recipe'] = cake_split['recipe:position'].str.split(":").str[0]
cake_split['position'] = cake_split['recipe:position'].str.split(":").str[1]
cake_drop = cake_split.drop(columns='recipe:position')
cake_drop.reindex()
# %%
cake_drop.head()
#%%
id = 'recipe'
cake_melt = cake_drop.melt(id_vars=['recipe', 'position'])
cake_melt.reindex()
# %%
cake_melt.pivot(index=['recipe', 'position'], columns='variable')
# %%
cake_melt.info()
#%%
# c. the average best tastiness reciepe is reciepe b
cake_melt.groupby('recipe').value.mean().idxmax()
#%%
# d. oven temp on average best results is 275 degrees
cake_melt.groupby('variable').value.mean().idxmax()
#%%
# e. combo best is reciepe b (bottom) at 300 degrees
cake_melt[cake_melt.value == cake_melt.value.max()]
# %%
