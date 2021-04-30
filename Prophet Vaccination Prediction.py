#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpldatacursor import datacursor
import mplcursors


# In[2]:


url = 'https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/country_data/India.csv?raw=true'

df = pd.read_csv(url,parse_dates = True)
df.head()


# In[3]:


df.tail()


# In[4]:


vaccination = df[['date','total_vaccinations']]


# In[5]:


vaccination


# In[6]:


vaccination.columns = ['ds','y']
vaccination['ds'] = pd.to_datetime(vaccination['ds'])


# In[7]:


days_to_predict = 30
m = Prophet(interval_width=0.95)
m.fit(vaccination)
future = m.make_future_dataframe(periods=days_to_predict)
future_vaccination = future.copy() # for non-baseline predictions later on
future.tail(days_to_predict)


# In[8]:


forecast = m.predict(future)

forecast


# In[9]:


forecast[['ds', 'yhat','yhat_lower','yhat_upper']].tail(days_to_predict)


# In[10]:


plt.style.use('fivethirtyeight')


# In[11]:


from fbprophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)


# In[12]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[13]:


vaccination_forcast_plot=m.plot(forecast)
#plt.legend()
plt.xlabel("Days starting from Jan 16 2021")
plt.ylabel("Number of Vaccinations")
mplcursors.cursor(hover=True)
plt.legend(handles=[vaccination_forcast_plot])


# In[14]:


plot2=m.plot_components(forecast)

