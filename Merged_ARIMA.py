import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv',index_col='Date_YMD',parse_dates=True)
df['Active'] = df['Total Confirmed'] - (df['Total Deceased'] + df['Total Recovered'])
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
'''
# Fit auto_arima function to AirPassengers dataset
stepwise_fit = auto_arima(df['Active'], start_p = 1, start_q = 1,max_p = 3, max_q = 3, m = 12,start_P = 0
                          , seasonal = True,
d = None, D = 1, trace = True,
error_action ='ignore', # we don't want to know if an order does not work
suppress_warnings = True, # we don't want convergence warnings
stepwise = True)# set to stepwise

# To print the summary
'''

'''
train = df.iloc[:int(df.shape[0]*0.95)]
test = df.iloc[int(df.shape[0]*0.95):] # set one year(12 months) for testing

# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df['Active'],order = (0,2,1),seasonal_order =(1,1,1,12))

result = model.fit()
result.summary()
start = len(train)
end = len(train) + len(test) - 1

# Predictions for one-year against the test set
predictions = result.predict(start, end,typ = 'levels').rename("Predictions")

# plot predictions and actual values
predictions.plot(figsize = (5.5,3.6),legend = True)
test['Active'].plot(figsize = (5.5,3.6),legend = True)

# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Calculate root mean squared error
#rmse(test["Total Confirmed"], predictions)

# Calculate mean squared error
mean_squared_error(test["Active"], predictions)


# In[9]:


rmse(test["Active"], predictions)


# In[10]:


plt.style.use('fivethirtyeight')


# In[11]:


import mplcursors


# In[12]:


#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[13]:


# Train the model on the full dataset
model = SARIMAX(df['Active'],order = (0,2,1),seasonal_order =(0, 1, 1, 12))
result = model.fit()

# Forecast for the next 36 days
forecast = result.predict(start = len(df),end = (len(df)-1) + 3 * 12,typ = 'levels').rename('Forecast')

# Plot the forecast values
df['Active'].plot(figsize = (5.5,3.6), legend = True)
forecast.plot(legend = True)

mplcursors.cursor(hover=True)
plt.legend(prop={"size":8})
plt.title("Arima Model Prediction of active cases for next 36 Days",fontsize=12)
plt.grid(color = 'green', linestyle = '--', linewidth = 1, which = 'major')
plt.xlabel("Days starting from 30th January 2020",  fontsize=12)
plt.ylabel("Total Active",  fontsize=12)
plt.savefig('image/Arima_Active.png')

'''
# In[14]:


#ARIMA CONFIRMED


# In[ ]:





# In[15]:


'''
stepwise_fit = auto_arima(df['Total Confirmed'], start_p = 1, start_q = 1,max_p = 3, max_q = 3, m = 12,start_P = 0
                          , seasonal = True,d = None, D = 1, trace = True,
                          error_action ='ignore', # we don't want to know if an order does not work
                        suppress_warnings = True, # we don't want convergence warnings
                          stepwise = True)# set to stepwise

# To print the summary

'''


# In[16]:

'''
# Split data into train / test sets
train = df.iloc[:int(df.shape[0]*0.95)]
test = df.iloc[int(df.shape[0]*0.95):] # set one year(12 months) for testing

# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df['Total Confirmed'],order = (3,2,3),seasonal_order =(0,1,2,12))

result = model.fit()


# In[17]:


start = len(train)
end = len(train) + len(test) - 1

# Predictions for one-year against the test set
predictions = result.predict(start, end,typ = 'levels').rename("Predictions")

# plot predictions and actual values
predictions.plot(legend = True)
test['Total Confirmed'].plot(legend = True)


# In[18]:


# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Calculate root mean squared error
#rmse(test["Total Confirmed"], predictions)

# Calculate mean squared error
mean_squared_error(test["Total Confirmed"], predictions)


# In[19]:



rmse(test["Total Confirmed"], predictions)


# In[20]:


plt.style.use('fivethirtyeight')


# In[21]:


import mplcursors


# In[22]:


#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[23]:


model = model = SARIMAX(df['Total Confirmed'],order = (0,2,1),seasonal_order =(0,1,1,12))
result = model.fit()

# Forecast for the next 3 years
forecast = result.predict(start = len(df),end = (len(df)-1) + 3 * 12,typ = 'levels').rename('Forecast')

df['Total Confirmed'].plot(figsize = (5.5,3.6), legend = True)
forecast.plot(legend = True)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
mplcursors.cursor(hover=True)
plt.legend(prop={"size":8})
plt.title("Arima Model Prediction of Confirmed cases for next 36 Days",fontsize=12)
plt.xlabel("Days starting from 30th January 2020",  fontsize=12)
plt.ylabel("Total Confirmed",  fontsize=12)
plt.savefig('image/Arima_Confirmed.png')

'''
# In[24]:


#ARIMA DECEASED


# In[ ]:





# In[25]:


'''
# Fit auto_arima function to AirPassengers dataset
stepwise_fit = auto_arima(df['Total Deceased'], start_p = 1, start_q = 1,max_p = 3, max_q = 3, m = 12,start_P = 0
                          , seasonal = True,
d = None, D = 1, trace = True,
error_action ='ignore', # we don't want to know if an order does not work
suppress_warnings = True, # we don't want convergence warnings
stepwise = True)# set to stepwise

# To print the summary
stepwise_fit.summary()
'''


# In[26]:
'''

# Split data into train / test sets
train = df.iloc[:int(df.shape[0]*0.95)]
test = df.iloc[int(df.shape[0]*0.95):] # set one year(12 months) for testing

# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df['Total Deceased'],order = (0,2,1),seasonal_order =(0,1,1,12))

result = model.fit()
result.summary()


# In[27]:


start = len(train)
end = len(train) + len(test) - 1

# Predictions for one-year against the test set
predictions = result.predict(start, end,typ = 'levels').rename("Predictions")

# plot predictions and actual values
predictions.plot(legend = True)
test['Total Deceased'].plot(legend = True)


# In[28]:


# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Calculate root mean squared error
#rmse(test["Total Deceased"], predictions)

# Calculate mean squared error
mean_squared_error(test["Total Deceased"], predictions)


# In[29]:


rmse(test["Total Deceased"], predictions)


# In[30]:


plt.style.use('fivethirtyeight')


# In[31]:


import mplcursors


# In[32]:


#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[33]:


# Train the model on the full dataset
model = model = SARIMAX(df['Total Deceased'],order = (0,2,1),seasonal_order =(0, 1, 1, 12))
result = model.fit()

# Forecast for the next 3 years
forecast = result.predict(start = len(df),end = (len(df)-1) + 3 * 12,typ = 'levels').rename('Forecast')

# Plot the forecast values
df['Total Deceased'].plot(figsize = (5.5,3.6), legend = True)
forecast.plot(legend = True)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
mplcursors.cursor(hover=True)
plt.legend(prop={"size":8})
plt.title("Arima Model Prediction of Deceased cases for next 36 Days",fontsize=12)
plt.xlabel("Days starting from 30th January 2020",  fontsize=12)
plt.ylabel("Total Deceased",  fontsize=12)
plt.savefig('image/Arima_Deaceased.png')

# In[34]:
'''

#ARIMA RECOVERED


# In[ ]:





# In[35]:


'''
# Fit auto_arima function to AirPassengers dataset
stepwise_fit = auto_arima(df['Total Recovered'], start_p = 1, start_q = 1,max_p = 3, max_q = 3, m = 12,start_P = 0
                          , seasonal = True,
d = None, D = 1, trace = True,
error_action ='ignore', # we don't want to know if an order does not work
suppress_warnings = True, # we don't want convergence warnings
stepwise = True)# set to stepwise

# To print the summary
stepwise_fit.summary()
'''


# In[36]:


# Split data into train / test sets
train = df.iloc[:int(df.shape[0]*0.95)]
test = df.iloc[int(df.shape[0]*0.95):] # set one year(12 months) for testing

# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df['Total Recovered'],order = (1,2,0),seasonal_order =(0,1,2,12))

result = model.fit()
result.summary()


# In[37]:


start = len(train)
end = len(train) + len(test) - 1

# Predictions for one-year against the test set
predictions = result.predict(start, end,typ = 'levels').rename("Predictions")

# plot predictions and actual values
predictions.plot(legend = True)
test['Total Recovered'].plot(legend = True)


# In[38]:


# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Calculate root mean squared error
#rmse(test["Total Recovered"], predictions)

# Calculate mean squared error
mean_squared_error(test["Total Recovered"], predictions)


# In[39]:


rmse(test["Total Recovered"], predictions)


# In[40]:


plt.style.use('fivethirtyeight')


# In[41]:


import mplcursors


# In[42]:


#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[43]:


# Train the model on the full dataset
model = model = SARIMAX(df['Total Recovered'],order = (0,2,2),seasonal_order =(0, 1, 1, 12))
result = model.fit()

# Forecast for the next 3 years
forecast = result.predict(start = len(df),end = (len(df)-1) + 3 * 12,typ = 'levels').rename('Forecast')

# Plot the forecast values
df['Total Recovered'].plot(figsize = (5.5,3.6), legend = True)
forecast.plot(legend = True)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
mplcursors.cursor(hover=True)
plt.legend(prop={"size":8})
plt.title("Arima Model Prediction of Recovered cases for next 36 Days",fontsize=12)
plt.xlabel("Days starting from 30th January 2020",  fontsize=12)
plt.ylabel("Total Recovered",  fontsize=12)
plt.savefig('image/Arima_Recovered.png')
#pyplot.legend()


# In[ ]:





# In[ ]:




