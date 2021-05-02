import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mplcursors
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import matplotlib.dates as mdates
from mpldatacursor import datacursor

# Read the AirPassengers dataset
url = 'https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/country_data/India.csv?raw=true'

df = pd.read_csv(url,parse_dates = True)#index_col ='date',parse_dates = True)
df1=df.drop(['location','vaccine','source_url','people_vaccinated','people_fully_vaccinated'],axis=1)
df1.date = pd.to_datetime(df1.date)
'''
stepwise_fit = auto_arima(df1['total_vaccinations'], start_p = 1, start_q = 1,max_p = 3, max_q = 3, m = 12,start_P = 0                          , seasonal = True,
d = None, D = 1, trace = True,
error_action ='ignore', # we don't want to know if an order does not work
suppress_warnings = True, # we don't want convergence warnings
stepwise = True)# set to stepwise

# To print the summary
stepwise_fit.summary()
'''
train = df1.iloc[:len(df)-24]#len(df)-24]int(df1.shape[0]*0.95
test = df1.iloc[len(df)-24:] # set one year(12 months) for testing
model = SARIMAX(df1['total_vaccinations'],order = (0,1,1),seasonal_order =(0,1,1,12))
result = model.fit()
result.summary()
start = len(train)
end = len(train) + len(test) - 1
predictions = result.predict(start, end,typ = 'levels').rename("Predictions")

# plot predictions and actual values
predictions.plot(legend = True)
test['total_vaccinations'].plot(legend = True)
mean_squared_error(test["total_vaccinations"], predictions)
rmse(test["total_vaccinations"], predictions)
plt.style.use('fivethirtyeight')
model = model = SARIMAX(df1['total_vaccinations'],order = (0,1,1),seasonal_order =(0,1,1,12))
result = model.fit()
forecast = result.predict(start = len(df1),end = (len(df1)-1) + 3 * 12,typ = 'levels').rename('Forecast')
df1['total_vaccinations'].plot(figsize = (11,5), legend = True)
forecast.plot(figsize = (11,5), legend = True)

plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
mplcursors.cursor(hover=True)
plt.legend(prop={"size":8})
plt.title("Arima Model Prediction of Vaccination for next 36 Days", fontsize=14)
plt.xlabel("Days starting from 16th January 2021",  fontsize=12)
plt.ylabel("Total vaccinations",  fontsize=12)
plt.savefig('image/Arima_Vacc.png')






#prophet Model


url = 'https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/country_data/India.csv?raw=true'

df = pd.read_csv(url,parse_dates = True)
vaccination = df[['date','total_vaccinations']]
vaccination.columns = ['ds','y']
vaccination['ds'] = pd.to_datetime(vaccination['ds'])
days_to_predict = 30
m = Prophet(interval_width=0.95)
m.fit(vaccination)
future = m.make_future_dataframe(periods=days_to_predict)
future_vaccination = future.copy() # for non-baseline predictions later on
future.tail(days_to_predict)
forecast = m.predict(future)
forecast[['ds', 'yhat','yhat_lower','yhat_upper']].tail(days_to_predict)
plt.style.use('fivethirtyeight')
plot_plotly(m, forecast)
vaccination_forcast_plot=m.plot(forecast,figsize = (11,5))
#plt.legend()
plt.legend(prop={"size":8})
plt.title("Prophet Model Prediction of Vaccinations for next 36 Days",  fontsize=12)
plt.xlabel("Days starting from Jan 16 2021",  fontsize=12)
plt.ylabel("Number of Vaccinations",  fontsize=12)
mplcursors.cursor(hover=True)
plt.legend(handles=[vaccination_forcast_plot])
plt.savefig('image/Prophet_Vacc.png')

