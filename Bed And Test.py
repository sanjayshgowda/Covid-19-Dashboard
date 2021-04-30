#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[1]:


import numpy as np
from numpy import inf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs.scatter.marker import Line
from plotly.graph_objs import Line
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')


# Global Overview

# In[2]:


worldwide = pd.read_csv('https://covid.ourworldindata.org/data/ecdc/total_cases.csv')


# In[3]:


worldwide['date'] = pd.to_datetime(worldwide['date'])
worldwide = worldwide.reset_index()
a = worldwide.iloc[worldwide.shape[0]-1]
a = list(a)[0]
top_10 = worldwide.tail(1)
top_10 = top_10.transpose()
top_10 = top_10.reset_index()
top_10 = top_10.rename(columns = {'index' : 'Country'})
top_10 = top_10.rename(columns = {a : 'Cases'})
top_10.drop([0, 1], inplace = True)
top_10 = top_10.reset_index()
top_10 = top_10[top_10['Country'] != 'World']
top_10.drop(columns = ['index'], inplace = True)
top_10 = top_10.sort_values(by = 'Cases', ascending = False).head(10)
top_10


# In[4]:


country_list = np.array(top_10['Country'])


# In[5]:


country_wise = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')


# In[6]:


country_wise = country_wise[country_wise['location'] != 'world']


# In[7]:


world = country_wise.sort_values(by = 'date')


# In[8]:


country_wise['date'] = pd.to_datetime(country_wise['date'])


# In[9]:


country_wise['Mortality_Rate'] = (country_wise['total_deaths']/country_wise['total_cases'])*100


# In[10]:


cases = []
for col in country_wise:
    for col in country_list:
        cases.append(country_wise.loc[country_wise['location'] == col])


# In[11]:


country_1 = cases[0]
country_2 = cases[1]
country_3 = cases[2]
country_4 = cases[3]
country_5 = cases[4]
country_6 = cases[5]
country_7 = cases[6]
country_8 = cases[7]
country_9 = cases[8]
country_10 = cases[9]


# In[12]:


hospital_beds_per_thousand = country_1.merge(country_2, on = 'date', how = 'left')


# In[13]:


hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'hospital_beds_per_thousand_x', 'hospital_beds_per_thousand_y']]


# In[14]:


hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand_x' : 'country_1', 'hospital_beds_per_thousand_y' : 'country_2'}, inplace = True)


# In[15]:


hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_3, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_3'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_4, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_4'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_5, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_5'}, inplace = True)


# In[16]:


hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_6, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_6'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_7, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_7'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_8, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_8'}, inplace = True)


# In[17]:


hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_9, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_9'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_10, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'country_9', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_10'}, inplace = True)


# In[18]:


hospital_beds_per_thousand = hospital_beds_per_thousand.fillna(0)


# In[19]:


hospital_beds_per_thousand = hospital_beds_per_thousand.reset_index()


# In[20]:


hospital_beds_per_thousand.rename(columns = {'index':'Days'}, inplace = True)


# In[21]:


beds = hospital_beds_per_thousand.tail(1)
beds.rename(columns = {'country_1' : country_list[0], 
                       'country_2' : country_list[1],
                       'country_3' : country_list[2],
                       'country_4' : country_list[3],
                       'country_5' : country_list[4],
                       'country_6' : country_list[5],
                       'country_7' : country_list[6],
                       'country_8' : country_list[7],
                       'country_9' : country_list[8],
                       'country_10' : country_list[9],},
            inplace = True)
a = int(beds['Days'])


# In[22]:


beds=beds.transpose()


# In[23]:


beds = beds.reset_index()


# In[24]:


beds.drop([0, 1], inplace = True)


# In[25]:


beds.rename(columns = {'index' : 'Country', a : 'Hospital_beds_per_thousand'}, inplace = True)


# In[26]:


beds_fig = px.bar(beds,
             x = 'Country',
             y = 'Hospital_beds_per_thousand',
             height = 500,
             title = 'Hospital Beds Per Thousand',
             color = 'Country')
beds_fig.show()


# In[27]:


country_1['test_per_confirmed'] = country_1['new_tests_smoothed']/country_1['new_cases']
country_2['test_per_confirmed'] = country_2['new_tests_smoothed']/country_2['new_cases']
country_3['test_per_confirmed'] = country_3['new_tests_smoothed']/country_3['new_cases']
country_4['test_per_confirmed'] = country_4['new_tests_smoothed']/country_4['new_cases']
country_5['test_per_confirmed'] = country_5['new_tests_smoothed']/country_5['new_cases']
country_6['test_per_confirmed'] = country_6['new_tests_smoothed']/country_6['new_cases']
country_7['test_per_confirmed'] = country_7['new_tests_smoothed']/country_7['new_cases']
country_8['test_per_confirmed'] = country_8['new_tests_smoothed']/country_8['new_cases']
country_9['test_per_confirmed'] = country_9['new_tests_smoothed']/country_9['new_cases']
country_10['test_per_confirmed'] = country_10['new_tests_smoothed']/country_10['new_cases']


# In[28]:


test_per_confirmed = country_1.merge(country_2, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'test_per_confirmed_x', 'test_per_confirmed_y']]
test_per_confirmed.rename(columns = {'test_per_confirmed_x' : 'country_1', 'test_per_confirmed_y' : 'country_2'}, inplace = True)


# In[29]:


test_per_confirmed = test_per_confirmed.merge(country_3, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'test_per_confirmed']]
test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_3'}, inplace = True)
test_per_confirmed = test_per_confirmed.merge(country_4, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'test_per_confirmed']]
test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_4'}, inplace = True)
test_per_confirmed = test_per_confirmed.merge(country_5, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'test_per_confirmed']]
test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_5'}, inplace = True)
test_per_confirmed = test_per_confirmed.merge(country_6, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'test_per_confirmed']]
test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_6'}, inplace = True)
test_per_confirmed = test_per_confirmed.merge(country_7, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'test_per_confirmed']]
test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_7'}, inplace = True)
test_per_confirmed = test_per_confirmed.merge(country_8, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'test_per_confirmed']]
test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_8'}, inplace = True)
test_per_confirmed = test_per_confirmed.merge(country_9, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'test_per_confirmed']]
test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_9'}, inplace = True)
test_per_confirmed = test_per_confirmed.merge(country_10, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'country_9', 'test_per_confirmed']]
test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_10'}, inplace = True)


# In[30]:


test_per_confirmed = test_per_confirmed.fillna(0)


# In[31]:


test_per_confirmed = test_per_confirmed.reset_index()


# In[32]:


test_per_confirmed.rename(columns = {'index':'Days'}, inplace = True)


# In[33]:


test_per_confirmed.set_index('date', inplace = True)
test_per_confirmed = test_per_confirmed.rolling(7).mean()


# In[34]:


test_per_confirmed.reset_index(inplace = True)
test_per_confirmed.rename(columns = {'index' : 'date'})


# In[35]:


test_per_confirmed = test_per_confirmed.fillna(0)


# In[36]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_1'],
                    mode='lines',
                    name=country_list[0]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_2'],
                    mode='lines',
                    name=country_list[1]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_3'],
                    mode='lines',
                    name=country_list[2]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_4'],
                    mode='lines',
                    name=country_list[3]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_5'],
                    mode='lines',
                    name=country_list[4]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_6'],
                    mode='lines', 
                    name=country_list[5]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_7'],
                    mode='lines',
                    name=country_list[6]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_8'],
                    mode='lines',
                    name=country_list[7]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_9'],
                    mode='lines', 
                    name=country_list[8]))
fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_10'],
                    mode='lines',
                    name=country_list[9]))
fig.update_layout(title = 'Test conducted per confirmed case (7 day rolling average)')
fig.update_xaxes(title= '------>Timeline' ,showline=False)
fig.update_yaxes(title= '------>Tests / Confirmed Case', showline=False)
fig.show()


# Impact of COVID-19 in india

# In[37]:


raw_data_1 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data1.csv')
raw_data_2 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data2.csv')
raw_data_3 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data3.csv')
raw_data_4 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data4.csv')
raw_data_5 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data5.csv')
status_1 = pd.read_csv('https://api.covid19india.org/csv/latest/death_and_recovered1.csv')
status_2 = pd.read_csv('https://api.covid19india.org/csv/latest/death_and_recovered2.csv')
statewise = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv')
case_time_series = pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
districtwise = pd.read_csv('https://api.covid19india.org/csv/latest/district_wise.csv')
statewise_tested = pd.read_csv('https://api.covid19india.org/csv/latest/statewise_tested_numbers_data.csv')
icmr_tested = pd.read_csv('https://api.covid19india.org/csv/latest/tested_numbers_icmr_data.csv')


# In[38]:


total = pd.DataFrame(statewise[statewise['State'] == 'Total'])


# In[39]:


total = total[['Recovered', 'Deaths', 'Active']]
total = total.transpose()
total = total.reset_index()
total.rename(columns = {'index' : 'Property', 0 : 'Numbers'}, inplace = True)
total


# In[40]:


fig_total = px.pie(total, 
                  values = 'Numbers', 
                  names = 'Property', 
                  title = 'Current COVID-19 Status',
                  color_discrete_map={'Active':'blue',
                                 'Recovered':'green',
                                 'Deaths':'red'})
fig_total.show()


# In[41]:


statewise


# In[42]:


statewise=statewise.drop([0])


# In[43]:


df_confirmed = statewise[['State', 'Confirmed']]
df_recovered = statewise[['State', 'Recovered']]
df_death = statewise[['State', 'Deaths']]
df_active = statewise[['State', 'Active']]


# In[44]:


fig_statewise = make_subplots(rows=2, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}], [{"type": "pie"}, {"type": "pie"}]])

fig_statewise.add_trace(go.Pie(
     values=df_confirmed['Confirmed'],
     labels=df_confirmed['State'],
     domain=dict(x=[0, 0.5]),
     title_text="Confirmed Cases"), 
     row=1, col=1)

fig_statewise.add_trace(go.Pie(
     values=df_active['Active'],
     labels=df_active['State'],
     domain=dict(x=[0.5, 1.0]),
     title_text="Active Cases"),
    row=1, col=2)
fig_statewise.add_trace(go.Pie(
     values=df_recovered['Recovered'],
     labels=df_recovered['State'],
     domain=dict(x=[0, 0.5]),
     title_text="Recovered"),
    row=2, col=1)

fig_statewise.add_trace(go.Pie(
     values=df_death['Deaths'],
     labels=df_death['State'],
     domain=dict(x=[0.5, 1.0]),
     title_text="Deaths"),
    row=2, col=2)

fig_statewise.update_traces(hoverinfo='label+percent+name', textinfo='none')
fig_statewise.show()


# In[45]:


statewise_5k = statewise[statewise['Confirmed'] >= 5000]
statewise_5k = statewise_5k[statewise_5k['State'] != 'State Unassigned']
statewise_5k = statewise_5k[statewise_5k['State'] != 'Total']


# In[46]:


statewise_5k


# In[47]:


fig = go.Figure(data=[
    go.Bar(name='Confirmed', x=statewise_5k['State'], y=statewise_5k['Confirmed'], marker_color = 'Blue'),
    go.Bar(name='Recovered', x=statewise_5k['State'], y=statewise_5k['Recovered'], marker_color = 'Green'),
    go.Bar(name='Deceased', x=statewise_5k['State'], y=statewise_5k['Deaths'], marker_color = 'Red')
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# In[48]:


net_updated = statewise[['State', 'Confirmed', 'Active', 'Recovered', 'Deaths']]
net_updated['Mortaliy Rate'] = net_updated['Deaths']/net_updated['Confirmed']
net_updated['Recovery Rate'] = net_updated['Recovered']/net_updated['Confirmed']
net_updated = net_updated.fillna(0)
net_updated.style.background_gradient(cmap = 'Reds')


# In[49]:


beds = pd.read_json('https://api.rootnet.in/covid19-in/hospitals/beds.json')
beds = pd.DataFrame(beds['data']['regional'])


# In[50]:


statewise_beds = beds[['state', 'totalBeds']]
statewise_beds = statewise_beds[statewise_beds['state'] != 'INDIA']
statewise_beds = statewise_beds.sort_values(by = 'totalBeds', ascending = False)
statewise_beds.style.background_gradient(cmap='Greens')


# In[51]:


fig = px.bar(statewise_beds, 
            x = 'totalBeds',
            y = 'state', 
            orientation = 'h',
            title = 'Hospital beds in each state', 
            color = 'state')
fig.show()


# In[52]:


urban_rural_beds = beds[['state', 'urbanHospitals','ruralHospitals']]
urban_rural_beds = urban_rural_beds[urban_rural_beds['state'] != 'INDIA']
urban_rural_beds.style.background_gradient(cmap='Greys')


# In[53]:


fig = go.Figure(data=[
    go.Bar(name='Urban Hospitals', x=urban_rural_beds['state'], y=urban_rural_beds['urbanHospitals']),
    go.Bar(name='Rural Hospitals', x=urban_rural_beds['state'], y=urban_rural_beds['ruralHospitals'])
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# In[54]:


urban_rural_beds = beds[['state', 'urbanBeds','ruralBeds']]
urban_rural_beds = urban_rural_beds[urban_rural_beds['state'] != 'INDIA']
urban_rural_beds.style.background_gradient(cmap='Blues')


# In[55]:


fig = go.Figure(data=[
    go.Bar(name='Urban Beds', x=urban_rural_beds['state'], y=urban_rural_beds['urbanBeds']),
    go.Bar(name='Rural Beds', x=urban_rural_beds['state'], y=urban_rural_beds['ruralBeds'])
])
fig.update_layout(barmode='group')
fig.show()


# In[56]:


states = districtwise['State'].unique()
state = pd.DataFrame()
i = 1
for col in states:
    while(i < len(states)):
        x = districtwise[districtwise['State'] == states[i]].sort_values(by = 'Confirmed', ascending = False).head(5)[['State', 'District', 'Confirmed']]
        state = pd.concat([state, x])
        i = i+1
state = state[(state['District'] != 'Unknown')]
state.style.background_gradient(cmap = 'Reds')


# In[57]:


trend = case_time_series[['Date', 'Total Confirmed', 'Total Recovered', 'Total Deceased']]
trend['Total Active'] = trend['Total Confirmed'] - (trend['Total Recovered'] + trend['Total Deceased'])


# In[58]:


fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=trend['Date'], 
    y=trend['Total Confirmed'],
    mode = 'lines+markers',
    name = 'Confirmed'
    ))
fig_trend.add_trace(go.Scatter(
    x=trend['Date'], 
    y=trend['Total Active'],
    mode = 'lines+markers',
    name = 'Active'
))
fig_trend.add_trace(go.Scatter(
    x=trend['Date'],
    y=trend['Total Recovered'],
    mode = 'lines+markers',
    name = 'Recovered'
))
fig_trend.add_trace(go.Scatter(
    x=trend['Date'],
    y=trend['Total Deceased'],
    mode = 'lines+markers',
    name = 'Deceased'
))

fig_trend.update_layout(title = 'Trends', showlegend=True)
fig_trend.update_xaxes(title = '------>Timeline', showline = False)
fig_trend.update_yaxes(title = '------>Numbers', showline = False)
fig_trend.show()


# In[59]:


samples_tested = icmr_tested[['Update Time Stamp', 'Total Samples Tested']]
samples_tested = samples_tested.set_index('Update Time Stamp')
samples_tested = samples_tested.diff()
samples_tested = samples_tested.reset_index()
samples_tested['Update Time Stamp'] = pd.to_datetime(samples_tested['Update Time Stamp'])
samples_tested['Update Time Stamp'] = samples_tested['Update Time Stamp'].dt.strftime('%d-%m-%Y')
samples_tested['Date'] = pd.DatetimeIndex(samples_tested['Update Time Stamp']).date
samples_tested['Month'] = pd.DatetimeIndex(samples_tested['Update Time Stamp']).month
samples_tested = samples_tested[['Date', 'Month', 'Total Samples Tested']]
samples_tested = samples_tested.fillna(0)


# In[60]:


fig_daily_tested = px.scatter(samples_tested,
                          x = 'Date',
                          y = 'Total Samples Tested',
                          color = 'Month',
                          hover_data = ['Date', 'Total Samples Tested'],
                          size = 'Total Samples Tested',
                          title = 'Number of Samples Tested Daily')
fig_daily_tested.show()

