
import numpy as np
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpldatacursor import datacursor
import mplcursors

df = pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
confirmed = df[['Date_YMD','Total Confirmed']]
confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
days_to_predict = 30
m = Prophet(interval_width=0.95)
m.fit(confirmed)
future = m.make_future_dataframe(periods=days_to_predict)
future_confirmed = future.copy() # for non-baseline predictions later on
future.tail(days_to_predict)
forecast = m.predict(future)
forecast[['ds', 'yhat','yhat_lower','yhat_upper']].tail(days_to_predict)
plt.style.use('fivethirtyeight')
mplcursors.cursor(hover=True)
from fbprophet.plot import plot_plotly, plot_components_plotly
plot_plotly(m, forecast)
confirmed_forcast_plot=m.plot(forecast,figsize = (11,5))
#plt.legend()
plt.legend(prop={"size":8})
plt.title("Prophet Model Prediction of Confirmed cases for next 36 Days",  fontsize=12)
plt.xlabel("Days starting from Jan 30 2020",  fontsize=12)
plt.ylabel("Number of Confirmed Cases",  fontsize=12)
mplcursors.cursor(hover=True)
plt.legend(handles=[confirmed_forcast_plot])
plt.savefig('image/Prophet_Confirmed.png')
plot2=m.plot_components(forecast)





from fbprophet.diagnostics import cross_validation
#df_cv = cross_validation(m,period='30 days', horizon = '40 days')
#df_cv = cross_validation(m,initial='150 days', period='30 days', horizon = '150 days')
df_cv = cross_validation(m, horizon = '30 days')#,initial='40 days', period='30 days',)
df['Active'] = df['Total Confirmed'] - (df['Total Deceased'] + df['Total Recovered'])
active = df[['Date_YMD','Active']]
active.columns = ['ds','y']
active['ds'] = pd.to_datetime(active['ds'])
days_to_predict = 30
m = Prophet(interval_width=0.95)
m.fit(active)
future = m.make_future_dataframe(periods=days_to_predict)
future_active = future.copy() # for non-baseline predictions later on
future.tail(days_to_predict)
forecast = m.predict(future)
forecast[['ds', 'yhat','yhat_lower','yhat_upper']].tail(days_to_predict)
plt.style.use('fivethirtyeight')
mplcursors.cursor(hover=True)
active_forcast_plot=m.plot(forecast,figsize = (11,5))
plt.legend()
plt.legend(prop={"size":8})
plt.title("Prophet Model Prediction of Active cases for next 36 Days",  fontsize=12)
plt.xlabel("Days starting from Jan 30 2020",  fontsize=12)
plt.ylabel("Number of Active Cases",  fontsize=12)
mplcursors.cursor(hover=True)
plt.legend(handles=[active_forcast_plot])
plt.savefig('image/Prophet_Active.png')









deceased = df[['Date_YMD','Total Deceased']]
deceased.columns = ['ds','y']
deceased['ds'] = pd.to_datetime(deceased['ds'])
days_to_predict = 30
m = Prophet(interval_width=0.95)
m.fit(deceased)
future = m.make_future_dataframe(periods=days_to_predict)
future_deceased = future.copy() # for non-baseline predictions later on
future.tail(days_to_predict)
forecast = m.predict(future)
forecast[['ds', 'yhat','yhat_lower','yhat_upper']].tail(days_to_predict)
plt.style.use('fivethirtyeight')
mplcursors.cursor(hover=True)
death_forcast_plot=m.plot(forecast,figsize = (11,5))
plt.legend()

plt.legend(prop={"size":8})
plt.title("Prophet Model Prediction of Death cases for next 36 Days",  fontsize=12)
plt.xlabel("Days starting from Jan 30 2020",  fontsize=12)
plt.ylabel("Number of Death Cases",  fontsize=12)
mplcursors.cursor(hover=True)
plt.legend(handles=[death_forcast_plot])
plt.savefig('image/Prophet_Death.png')











recovered = df[['Date_YMD','Total Recovered']]
recovered.columns = ['ds','y']
recovered['ds'] = pd.to_datetime(recovered['ds'])
days_to_predict = 30
m = Prophet(interval_width=0.95)
m.fit(recovered)
future = m.make_future_dataframe(periods=days_to_predict)
future_recovered = future.copy() # for non-baseline predictions later on
future.tail(days_to_predict)
forecast = m.predict(future)
forecast[['ds', 'yhat','yhat_lower','yhat_upper']].tail(days_to_predict)
plt.style.use('fivethirtyeight')
mplcursors.cursor(hover=True)
recovered_forcast_plot=m.plot(forecast,figsize = (11,5))
plt.legend()

plt.legend(prop={"size":8})
plt.title("Prophet Model Prediction of Recovered cases for next 36 Days",  fontsize=12)
plt.xlabel("Days starting from Jan 30 2020",  fontsize=12)
plt.ylabel("Number of Recovered Cases",  fontsize=12)
mplcursors.cursor(hover=True)
plt.legend(handles=[recovered_forcast_plot])
plt.savefig('image/Prophet_Recovered.png')
