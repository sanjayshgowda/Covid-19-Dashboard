import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs.scatter.marker import Line
from plotly.graph_objs import Line
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from datetime import date
import calendar
# data visualization library 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpldatacursor import datacursor
import mplcursors

df = pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')


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

#crs = mplcursors.cursor(ax,hover=True)
mplcursors.cursor(hover=True)
from fbprophet.plot import plot_plotly, plot_components_plotly

pro_active=plot_plotly(m, forecast)


app = dash.Dash(__name__, )

app.layout= html.Div([
    dcc.Graph(id= 'matplotlib-graph', figure=pro_active)


])

if __name__ == '__main__':
    app.run_server(host="localhost", port=8000, debug=True)