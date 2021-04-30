#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Read the AirPassengers dataset
url = 'https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/country_data/India.csv?raw=true'

df = pd.read_csv(url,parse_dates = True)#index_col ='date',parse_dates = True)

# Print the first five rows of the dataset
df.head()

# ETS Decomposition
#result = seasonal_decompose(df['Total Confirmed'],model ='multiplicative')

# ETS plot
#result.plot()


# In[2]:


df.tail()


# In[3]:


df1=df.drop(['location','vaccine','source_url','people_vaccinated','people_fully_vaccinated'],axis=1)


# In[4]:


df1.date = pd.to_datetime(df1.date)


# In[5]:


df1


# In[6]:


# To install the library
#pip install pmdarima

# Import the library
from pmdarima import auto_arima

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Fit auto_arima function to AirPassengers dataset
stepwise_fit = auto_arima(df1['total_vaccinations'], start_p = 1, start_q = 1,max_p = 3, max_q = 3, m = 12,start_P = 0
                          , seasonal = True,
d = None, D = 1, trace = True,
error_action ='ignore', # we don't want to know if an order does not work
suppress_warnings = True, # we don't want convergence warnings
stepwise = True)# set to stepwise

# To print the summary
stepwise_fit.summary()


# In[7]:


# Split data into train / test sets
train = df1.iloc[:len(df)-24]#len(df)-24]int(df1.shape[0]*0.95
test = df1.iloc[len(df)-24:] # set one year(12 months) for testing

# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df1['total_vaccinations'],order = (0,1,1),seasonal_order =(0,1,1,12))

result = model.fit()
result.summary()


# In[8]:


start = len(train)
end = len(train) + len(test) - 1

# Predictions for one-year against the test set
predictions = result.predict(start, end,typ = 'levels').rename("Predictions")

# plot predictions and actual values
predictions.plot(legend = True)
test['total_vaccinations'].plot(legend = True)


# In[9]:


# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Calculate root mean squared error
#rmse(test["Total Confirmed"], predictions)

# Calculate mean squared error
mean_squared_error(test["total_vaccinations"], predictions)


# In[10]:


rmse(test["total_vaccinations"], predictions)


# In[11]:


plt.style.use('fivethirtyeight')


# In[12]:


import mplcursors


# In[13]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[14]:


# Train the model on the full dataset
model = model = SARIMAX(df1['total_vaccinations'],order = (0,1,1),seasonal_order =(0,1,1,12))
result = model.fit()

# Forecast for the next 3 years
forecast = result.predict(start = len(df1),end = (len(df1)-1) + 3 * 12,typ = 'levels').rename('Forecast')

# Plot the forecast values
df1['total_vaccinations'].plot(figsize = (12, 5), legend = True)
forecast.plot(legend = True)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
mplcursors.cursor(hover=True)
plt.xlabel("Days starting from 16th January 2021")
plt.ylabel("Total vaccinations")
#pyplot.legend()


# In[ ]:




