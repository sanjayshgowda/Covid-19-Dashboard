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
import base64
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

# Boostrap CSS and font awesome . Option 1) Run from codepen directly Option 2) Copy css file to assets folder and run locally
external_stylesheets = []

#Insert your javascript here. In this example, addthis.com has been added to the web app for people to share their webpage
external_scripts = [{
        'type': 'text/javascript', #depends on your application
        'src': 'insert your addthis.com js here',
    }]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts = external_scripts)
app.title = 'Covid19 Dashboard '

#for heroku to run correctly
server = app.server

#Overwrite your CSS setting by including style locally
colors = {
    'background': '#2D2D2D',
    'text': '#E1E2E5',
    'figure_text': '#ffffff',
    'confirmed_text':'#3CA4FF',
    'deaths_text':'#f44336',
    'recovered_text':'#5A9E6F',
    'highest_case_bg':'#393939',
    
}

#Creating custom style for local use
divBorderStyle = {
    'backgroundColor' : '#393939',
    'borderRadius': '12px',
    'lineHeight': 0.9,
}

#Creating custom style for local use
boxBorderStyle = {
    'borderColor' : '#393939',
    'borderStyle': 'solid',
    'borderRadius': '10px',
    'borderWidth':2,
}

# Retrieve data


# get data directly from github. The data source provided by Johns Hopkins University.
url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'



# Data can also be saved locally and read from local drive
"""url_confirmed = 'time_series_covid19_confirmed_global.csv'
url_deaths = 'time_series_covid19_deaths_global.csv'
url_recovered = 'time_series_covid19_recovered_global.csv'"""

df_confirmed = pd.read_csv(url_confirmed)
df_deaths = pd.read_csv(url_deaths)
df_recovered = pd.read_csv(url_recovered)


def df_move1st_sg(df_t):

    #Moving India to the first row in the datatable
    df_t["new"] = range(1,len(df_t)+1)
    df_t.loc[df_t[df_t['Country/Region'] == 'India'].index.values,'new'] = 0
    df_t = df_t.sort_values("new").drop('new', axis=1)
    return df_t

# Data preprocessing for getting useful data and shaping data compatible to plotly plot

# Total cases
df_confirmed_total = df_confirmed.iloc[:, 4:].sum(axis=0)
df_deaths_total = df_deaths.iloc[:, 4:].sum(axis=0)
df_recovered_total = df_recovered.iloc[:, 4:].sum(axis=0)

# modified deaths dataset for mortality rate calculation
df_deaths_confirmed=df_deaths.copy()
df_deaths_confirmed['confirmed'] = df_confirmed.iloc[:,-1]

#Sorted - df_deaths_confirmed_sorted is different from others, as it is only modified later. Careful of it dataframe structure
df_deaths_confirmed_sorted = df_deaths_confirmed.sort_values(by=df_deaths_confirmed.columns[-2], ascending=False)[['Country/Region',df_deaths_confirmed.columns[-2],df_deaths_confirmed.columns[-1]]]
df_recovered_sorted = df_recovered.sort_values(by=df_recovered.columns[-1], ascending=False)[['Country/Region',df_recovered.columns[-1]]]
df_confirmed_sorted = df_confirmed.sort_values(by=df_confirmed.columns[-1], ascending=False)[['Country/Region',df_confirmed.columns[-1]]]

#Single day increase
df_deaths_confirmed_sorted['24hr'] = df_deaths_confirmed_sorted.iloc[:,-2] - df_deaths.sort_values(by=df_deaths.columns[-1], ascending=False)[df_deaths.columns[-2]]
df_recovered_sorted['24hr'] = df_recovered_sorted.iloc[:,-1] - df_recovered.sort_values(by=df_recovered.columns[-1], ascending=False)[df_recovered.columns[-2]]
df_confirmed_sorted['24hr'] = df_confirmed_sorted.iloc[:,-1] - df_confirmed.sort_values(by=df_confirmed.columns[-1], ascending=False)[df_confirmed.columns[-2]]

#Aggregate the countries with different province/state together
df_deaths_confirmed_sorted_total = df_deaths_confirmed_sorted.groupby('Country/Region').sum()
df_deaths_confirmed_sorted_total=df_deaths_confirmed_sorted_total.sort_values(by=df_deaths_confirmed_sorted_total.columns[0], ascending=False).reset_index()
df_recovered_sorted_total = df_recovered_sorted.groupby('Country/Region').sum()
df_recovered_sorted_total=df_recovered_sorted_total.sort_values(by=df_recovered_sorted_total.columns[0], ascending=False).reset_index()
df_confirmed_sorted_total = df_confirmed_sorted.groupby('Country/Region').sum()
df_confirmed_sorted_total=df_confirmed_sorted_total.sort_values(by=df_confirmed_sorted_total.columns[0], ascending=False).reset_index()

#Modified recovery csv due to difference in number of rows. Recovered will match ['Province/State','Country/Region']column with Confirmed ['Province/State','Country/Region']
df_recovered['Province+Country'] = df_recovered[['Province/State','Country/Region']].fillna('nann').agg('|'.join,axis=1)
df_confirmed['Province+Country'] = df_confirmed[['Province/State','Country/Region']].fillna('nann').agg('|'.join,axis=1)
df_recovered_fill = df_recovered
df_recovered_fill.set_index("Province+Country")
df_recovered_fill.set_index("Province+Country").reindex(df_confirmed['Province+Country'])
df_recovered_fill = df_recovered_fill.set_index("Province+Country").reindex(df_confirmed['Province+Country']).reset_index()
#split Province+Country back into its respective columns
new = df_recovered_fill["Province+Country"].str.split("|", n = 1, expand = True)
df_recovered_fill['Province/State']=new[0]
df_recovered_fill['Country/Region']=new[1]
df_recovered_fill['Province/State'].replace('nann','NaN')
#drop 'Province+Country' for all dataset
df_confirmed.drop('Province+Country',axis=1,inplace=True)
df_recovered.drop('Province+Country',axis=1,inplace=True)
df_recovered_fill.drop('Province+Country',axis=1,inplace=True)

# Data preprocessing for times series countries graph display 
# create temp to store sorting arrangement for all confirm, deaths and recovered.
df_confirmed_sort_temp = df_confirmed.sort_values(by=df_confirmed.columns[-1], ascending=False)

df_confirmed_t = df_move1st_sg(df_confirmed_sort_temp)
df_confirmed_t['Province+Country'] = df_confirmed_t[['Province/State','Country/Region']].fillna('nann').agg('|'.join,axis=1)
df_confirmed_t=df_confirmed_t.drop(['Province/State','Country/Region','Lat','Long'],axis=1).T

df_deaths_t = df_deaths.reindex(df_confirmed_sort_temp.index)
df_deaths_t = df_move1st_sg(df_deaths_t)
df_deaths_t['Province+Country'] = df_deaths_t[['Province/State','Country/Region']].fillna('nann').agg('|'.join,axis=1)
df_deaths_t=df_deaths_t.drop(['Province/State','Country/Region','Lat','Long'],axis=1).T
# take note use reovered_fill df
df_recovered_t = df_recovered_fill.reindex(df_confirmed_sort_temp.index)
df_recovered_t = df_move1st_sg(df_recovered_t)
df_recovered_t['Province+Country'] = df_recovered_t[['Province/State','Country/Region']].fillna('nann').agg('|'.join,axis=1)
df_recovered_t=df_recovered_t.drop(['Province/State','Country/Region','Lat','Long'],axis=1).T

df_confirmed_t.columns = df_confirmed_t.iloc[-1]
df_confirmed_t = df_confirmed_t.drop('Province+Country')

df_deaths_t.columns = df_deaths_t.iloc[-1]
df_deaths_t = df_deaths_t.drop('Province+Country')

df_recovered_t.columns = df_recovered_t.iloc[-1]
df_recovered_t = df_recovered_t.drop('Province+Country')

df_confirmed_t.index=pd.to_datetime(df_confirmed_t.index)
df_deaths_t.index=pd.to_datetime(df_confirmed_t.index)
df_recovered_t.index=pd.to_datetime(df_confirmed_t.index)

# Highest 10 plot data preprocessing
# getting highest 10 countries with confirmed case
name = df_confirmed_t.columns.str.split("|", 1)
df_confirmed_t_namechange=df_confirmed_t.copy()
# name0 = [x[0] for x in name]
name1 = [x[1] for x in name]
df_confirmed_t_namechange.columns = name1
df_confirmed_t_namechange=df_confirmed_t_namechange.groupby(df_confirmed_t_namechange.columns,axis=1).sum()
df_confirmed_t_namechange10 = df_confirmed_t_namechange.sort_values(by=df_confirmed_t_namechange.index[-1], axis=1, ascending=False).iloc[:,:10]
df_confirmed_t_stack = df_confirmed_t_namechange10.stack()
df_confirmed_t_stack=df_confirmed_t_stack.reset_index(level=[0,1])
df_confirmed_t_stack.rename(columns={"level_0": "Date",'level_1':'Countries', 0: "Confirmed"}, inplace=True)
# getting highest 10 countries with deceased case
name = df_deaths_t.columns.str.split("|", 1)
df_deaths_t_namechange=df_deaths_t.copy()
# name0 = [x[0] for x in name]
name1 = [x[1] for x in name]
df_deaths_t_namechange.columns = name1
df_deaths_t_namechange=df_deaths_t_namechange.groupby(df_deaths_t_namechange.columns,axis=1).sum()
df_deaths_t_namechange10 = df_deaths_t_namechange.sort_values(by=df_deaths_t_namechange.index[-1], axis=1, ascending=False).iloc[:,:10]
df_deaths_t_stack = df_deaths_t_namechange10.stack()
df_deaths_t_stack=df_deaths_t_stack.reset_index(level=[0,1])
df_deaths_t_stack.rename(columns={"level_0": "Date",'level_1':'Countries', 0: "Deceased"}, inplace=True)

# Recreate required columns for map data
map_data = df_confirmed[["Province/State", "Country/Region", "Lat", "Long"]]
map_data['Confirmed'] = df_confirmed.loc[:, df_confirmed.columns[-1]]
map_data['Deaths'] = df_deaths.loc[:, df_deaths.columns[-1]]
map_data['Recovered'] = df_recovered_fill.loc[:, df_recovered_fill.columns[-1]]
map_data['Recovered']=map_data['Recovered'].fillna(0).astype(int) #too covert value back to int and fillna with zero
#last 24 hours increase
map_data['Deaths_24hr']=df_deaths.iloc[:,-1] - df_deaths.iloc[:,-2]
map_data['Recovered_24hr']=df_recovered_fill.iloc[:,-1] - df_recovered_fill.iloc[:,-2]
map_data['Confirmed_24hr']=df_confirmed.iloc[:,-1] - df_confirmed.iloc[:,-2]
map_data.sort_values(by='Confirmed', ascending=False,inplace=True)
#Moving India to the first row in the datatable
map_data["new"] = range(1,len(map_data)+1)
map_data.loc[map_data[map_data['Country/Region'] == 'India'].index.values,'new'] = 0
map_data = map_data.sort_values("new").drop('new', axis=1)



# mapbox_access_token keys, not all mapbox function require token to function. 
mapbox_access_token = 'pk.eyJ1Ijoic2FuamF5c2hnb3dkYSIsImEiOiJja2lidHJ3NzMwNHRwMnFwZXFubHRzcWRrIn0.vE3X69-WtlhXCqfOhNC4dA'


# functions to create map


def gen_map(map_data,zoom,lat,lon):
    return {
        "data": [{
            "type": "scattermapbox",  #specify the type of data to generate, in this case, scatter map box is used
            "lat": list(map_data['Lat']),   #for markers location
            "lon": list(map_data['Long']),
            # "hoverinfo": "text",         
            "hovertext": [["Country/Region: {} <br>Province/State: {} <br>Confirmed: {} (+ {} past 24hrs)<br>Deaths: {} (+ {} past 24hrs)<br>Recovered: {} (+ {} past 24hrs)".format(i, j, k, k24, l, l24, m, m24)]
                          for i, j, k, l, m, k24, l24, m24 in zip(map_data['Country/Region'], map_data['Province/State'],
                                                   map_data['Confirmed'], map_data['Deaths'], map_data['Recovered'],
                                                    map_data['Confirmed_24hr'], map_data['Deaths_24hr'], map_data['Recovered_24hr'],)],

            "mode": "markers",
            "name": list(map_data['Country/Region']),
            "marker": {
                    "opacity": 0.7,
                    "size": np.log(map_data['Confirmed'])*4,
            }
        },
        
        ],
        "layout": dict(
            autosize=True,
            height=350,
            font=dict(color=colors['figure_text']),
            titlefont=dict(color=colors['text'], size='14'),
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            hovermode="closest",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            legend=dict(font=dict(size=10), orientation='h'),
            mapbox=dict(
                accesstoken=mapbox_access_token,
                style='mapbox://styles/mapbox/dark-v10',
                center=dict(
                    lon=lon,
                    lat=lat,
                ),
                zoom=zoom,
            )
        ),
    }


#Functions to create display for highest cases

def high_cases(countryname,total,single,color_word='#63b6ff',confirmed_total=1,deaths = False,):

    if deaths:

        percent = (total/confirmed_total)*100
        return html.P([ html.Span(countryname + ' | ' + f"{int(total):,d}",style={'backgroundColor': colors['highest_case_bg'], 'borderRadius': '6px',}),
                    html.Span(' +' + f"{int(single):,d}",
                             style={'color': color_word,'margin':2,'fontWeight': 'bold','fontSize': 14,}),
                    html.Span(f' ({percent:.2f}%)',
                             style={'color': color_word,'margin':2,'fontWeight': 'bold','fontSize': 14,}),
                   ],
                  style={
                        'textAlign': 'center',
                        'color': 'rgb(200,200,200)',
                        'fontsize':12,
                        }       
                )

    return html.P([ html.Span(countryname + ' | ' + f"{int(total):,d}",
                        style={'backgroundColor': colors['highest_case_bg'], 'borderRadius': '6px',}),
            html.Span(' +' + f"{int(single):,d}",
                        style={'color': color_word,'margin':2,'fontWeight': 'bold','fontSize': 14,}),
            ],
            style={
                'textAlign': 'center',
                'color': 'rgb(200,200,200)',
                'fontsize':12,
                }       
        )

#Convert datetime to Display datetime with following format - 06-Apr-2020

def datatime_convert(date_str,days_to_add=0):

    format_str = '%m/%d/%y' # The format
    datetime_obj = datetime.datetime.strptime(date_str, format_str)
    datetime_obj += datetime.timedelta(days=days_to_add)
    return datetime_obj.strftime('%d-%b-%Y')

def return_outbreakdays(date_str):
    format_str = '%d-%b-%Y' # The format
    datetime_obj = datetime.datetime.strptime(date_str, format_str).date()

    d0 = datetime.date(2020, 1, 22)
    delta = datetime_obj - d0
    return delta.days

noToDisplay = 8

confirm_cases = []
for i in range(noToDisplay):
    confirm_cases.append(high_cases(df_confirmed_sorted_total.iloc[i,0],df_confirmed_sorted_total.iloc[i,1],df_confirmed_sorted_total.iloc[i,2]))

deaths_cases = []
for i in range(noToDisplay):
    deaths_cases.append(high_cases(df_deaths_confirmed_sorted_total.iloc[i,0],df_deaths_confirmed_sorted_total.iloc[i,1],df_deaths_confirmed_sorted_total.iloc[i,3],'#ff3b4a',df_deaths_confirmed_sorted_total.iloc[i,2],True))

confirm_cases_24hrs = []
for i in range(noToDisplay):
    confirm_cases_24hrs.append(high_cases(df_confirmed_sorted_total.sort_values(by=df_confirmed_sorted_total.columns[-1], ascending=False).iloc[i,0],
                                            df_confirmed_sorted_total.sort_values(by=df_confirmed_sorted_total.columns[-1], ascending=False).iloc[i,1],
                                            df_confirmed_sorted_total.sort_values(by=df_confirmed_sorted_total.columns[-1], ascending=False).iloc[i,2],
                                            ))

deaths_cases_24hrs = []
for i in range(noToDisplay):
    deaths_cases_24hrs.append(high_cases(df_deaths_confirmed_sorted_total.sort_values(by=df_deaths_confirmed_sorted_total.columns[-1], ascending=False).iloc[i,0],
                                            df_deaths_confirmed_sorted_total.sort_values(by=df_deaths_confirmed_sorted_total.columns[-1], ascending=False).iloc[i,1],
                                            df_deaths_confirmed_sorted_total.sort_values(by=df_deaths_confirmed_sorted_total.columns[-1], ascending=False).iloc[i,3],
                                            '#ff3b4a',
                                            df_deaths_confirmed_sorted_total.sort_values(by=df_deaths_confirmed_sorted_total.columns[-1], ascending=False).iloc[i,2],
                                            True))

# Prepare plotly figure to attached to dcc component
# Global outbreak Plot 
# Change date index to datetimeindex and share x-axis with all the plot
def draw_global_graph(df_confirmed_total,df_deaths_total,df_recovered_total,graph_type='Total Cases'):
    df_confirmed_total.index = pd.to_datetime(df_confirmed_total.index)

    if graph_type == 'Daily Cases':
        df_confirmed_total = (df_confirmed_total - df_confirmed_total.shift(1)).drop(df_confirmed_total.index[0])
        df_deaths_total = (df_deaths_total - df_deaths_total.shift(1)).drop(df_deaths_total.index[0])
        df_recovered_total = (df_recovered_total - df_recovered_total.shift(1)).drop(df_recovered_total.index[0])

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_confirmed_total.index, y=df_confirmed_total,
                            mode='lines+markers',
                            name='Confirmed',
                            line=dict(color='#3372FF', width=2),
                            fill='tozeroy',))
    fig.add_trace(go.Scatter(x=df_confirmed_total.index, y=df_recovered_total,
                            mode='lines+markers',
                            name='Recovered',
                            line=dict(color='#33FF51', width=2),
                            fill='tozeroy',))
    fig.add_trace(go.Scatter(x=df_confirmed_total.index, y=df_deaths_total,
                            mode='lines+markers',
                            name='Deaths',
                            line=dict(color='#FF3333', width=2),
                            fill='tozeroy',))

    fig.update_layout(
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color=colors['figure_text'],
        ),
        legend=dict(
            x=0.02,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                    
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=5
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, 
                    r=0, 
                    t=0, 
                    b=0
                    ),
        height=300,

    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')
    return fig


# Function to plot Highest 10 countries cases
def draw_highest_10(df_confirmed_t_stack, df_deaths_t_stack, graphHigh10_type='Confirmed Cases'):

    if graphHigh10_type=='Confirmed Cases':
        fig = px.line(df_confirmed_t_stack, x="Date", y="Confirmed", color='Countries',
             color_discrete_sequence = px.colors.qualitative.Light24)
    else:
        fig = px.line(df_deaths_t_stack, x="Date", y="Deceased", color='Countries',
             title='Deceased cases', color_discrete_sequence = px.colors.qualitative.Light24)

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        font=dict(
            family="Courier New, monospace",
            size=14,
            color=colors['figure_text'],
        ),
        legend=dict(
            x=0.02,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=9,
                color=colors['figure_text']
            ),
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, 
                    r=0, 
                    t=0, 
                    b=0
                    ),
        height=300,

    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

    # fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')
    return fig


# Function to plot Single Country Scatter Plot


def draw_singleCountry_Scatter(df_confirmed_t,df_deaths_t,df_recovered_t,selected_row=0, daily_change=False):
    
    if daily_change:
        df_confirmed_t = (df_confirmed_t - df_confirmed_t.shift(1)).drop(df_confirmed_t.index[0])
        df_deaths_t = (df_deaths_t - df_deaths_t.shift(1)).drop(df_deaths_t.index[0])
        df_recovered_t = (df_recovered_t - df_recovered_t.shift(1)).drop(df_recovered_t.index[0])
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_confirmed_t.index, y=df_confirmed_t.iloc[:,selected_row],
                             mode='lines+markers',
                             name='Confirmed',
                             line=dict(color='#3372FF', width=2),
                             fill='tozeroy',))
    fig.add_trace(go.Scatter(x=df_recovered_t.index, y=df_recovered_t.iloc[:,selected_row],
                             mode='lines+markers',
                             name='Recovered',
                             line=dict(color='#33FF51', width=2),
                             fill='tozeroy',))
    fig.add_trace(go.Scatter(x=df_deaths_t.index, y=df_deaths_t.iloc[:,selected_row],
                             mode='lines+markers',
                             name='Deceased',
                             line=dict(color='#FF3333', width=2),
                             fill='tozeroy',))


    new = df_recovered_t.columns[selected_row].split("|", 1)
    if new[0] == 'nann':
        title = new[1]
    else:
        title = new[1] + " - " + new[0]

    fig.update_layout(
        title=title + ' (Total Cases)',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=5
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=65, b=0),
        height=350,

    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')

    return fig



# Function to plot Single Country Bar with scatter Plot


def draw_singleCountry_Bar(df_confirmed_t,df_deaths_t,df_recovered_t,selected_row=0,graph_line='Line Chart'):
    

    df_confirmed_t = (df_confirmed_t - df_confirmed_t.shift(1)).drop(df_confirmed_t.index[0])
    df_deaths_t = (df_deaths_t - df_deaths_t.shift(1)).drop(df_deaths_t.index[0])
    df_recovered_t = (df_recovered_t - df_recovered_t.shift(1)).drop(df_recovered_t.index[0])
        
    fig = go.Figure()
    if graph_line=='Line Chart':
        fig.add_trace(go.Bar(x=df_confirmed_t.index, y=df_confirmed_t.iloc[:,selected_row],
                            name='Confirmed',
                            marker_color='#3372FF'
                            ))
        fig.add_trace(go.Bar(x=df_recovered_t.index, y=df_recovered_t.iloc[:,selected_row],
                            name='Recovered',
                            marker_color='#33FF51'
                            ))
        fig.add_trace(go.Bar(x=df_deaths_t.index, y=df_deaths_t.iloc[:,selected_row],
                            name='Deceased',
                            marker_color='#FF3333'
                            ))
  
    else:
        fig.add_trace(go.Scatter(x=df_confirmed_t.index, y=df_confirmed_t.iloc[:,selected_row],
                                mode='lines+markers',
                                name='Confirmed',
                                line=dict(color='#3372FF', width=2),
                                fill='tozeroy',))
        fig.add_trace(go.Scatter(x=df_recovered_t.index, y=df_recovered_t.iloc[:,selected_row],
                                mode='lines+markers',
                                name='Recovered',
                                line=dict(color='#33FF51', width=2),
                                fill='tozeroy',))
        fig.add_trace(go.Scatter(x=df_deaths_t.index, y=df_deaths_t.iloc[:,selected_row],
                                mode='lines+markers',
                                name='Deceased',
                                line=dict(color='#FF3333', width=2),
                                fill='tozeroy',))

    new = df_recovered_t.columns[selected_row].split("|", 1)
    if new[0] == 'nann':
        title = new[1]
    else:
        title = new[1] + " - " + new[0]

    fig.update_layout(
        title=title + ' (Daily Cases)',
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=5
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=65, b=0),
        height=350,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')

    return fig

# Bed Data Computation

worldwide = pd.read_csv('https://covid.ourworldindata.org/data/ecdc/total_cases.csv')

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
country_list = np.array(top_10['Country'])
country_wise = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
country_wise = country_wise[country_wise['location'] != 'world']
world = country_wise.sort_values(by = 'date')
country_wise['date'] = pd.to_datetime(country_wise['date'])
country_wise['Mortality_Rate'] = (country_wise['total_deaths']/country_wise['total_cases'])*100
cases = []
for col in country_wise:
    for col in country_list:
        cases.append(country_wise.loc[country_wise['location'] == col])
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
hospital_beds_per_thousand = country_1.merge(country_2, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'hospital_beds_per_thousand_x', 'hospital_beds_per_thousand_y']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand_x' : 'country_1', 'hospital_beds_per_thousand_y' : 'country_2'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_3, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_3'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_4, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_4'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_5, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_5'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_6, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_6'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_7, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_7'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_8, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_8'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_9, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_9'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_10, on = 'date', how = 'left')
hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'country_9', 'hospital_beds_per_thousand']]
hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_10'}, inplace = True)
hospital_beds_per_thousand = hospital_beds_per_thousand.fillna(0)
hospital_beds_per_thousand = hospital_beds_per_thousand.reset_index()
hospital_beds_per_thousand.rename(columns = {'index':'Days'}, inplace = True)
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
beds=beds.transpose()
beds = beds.reset_index()
beds.drop([0, 1], inplace = True)
beds.rename(columns = {'index' : 'Country', a : 'Hospital_beds_per_thousand'}, inplace = True)
fig_bed_per_1000 = px.bar(beds,
             x = 'Country',
             y = 'Hospital_beds_per_thousand',
             height = 500,
             title = 'Hospital Beds Per Thousand',
             color = 'Country')
fig_bed_per_1000.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=14,
            color=colors['figure_text'],
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=9,
                color=colors['figure_text']
            ),
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, 
                    r=0, 
                    t=0, 
                    b=0
                    ),
        height=350
        

    )
fig_bed_per_1000.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
fig_bed_per_1000.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

# fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')
fig_bed_per_1000.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')


#test conducted per confirmed case 7 day roling average
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
test_per_confirmed = country_1.merge(country_2, on = 'date', how = 'left')
test_per_confirmed = test_per_confirmed[['date', 'test_per_confirmed_x', 'test_per_confirmed_y']]
test_per_confirmed.rename(columns = {'test_per_confirmed_x' : 'country_1', 'test_per_confirmed_y' : 'country_2'}, inplace = True)
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
test_per_confirmed = test_per_confirmed.fillna(0)
test_per_confirmed = test_per_confirmed.reset_index()
test_per_confirmed.rename(columns = {'index':'Days'}, inplace = True)
test_per_confirmed.set_index('date', inplace = True)
test_per_confirmed = test_per_confirmed.rolling(7).mean()
test_per_confirmed.reset_index(inplace = True)
test_per_confirmed.rename(columns = {'index' : 'date'})
test_per_confirmed = test_per_confirmed.fillna(0)
fig_rolling7 = go.Figure()
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_1'],
                    mode='lines',
                    name=country_list[0]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_2'],
                    mode='lines',
                    name=country_list[1]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_3'],
                    mode='lines',
                    name=country_list[2]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_4'],
                    mode='lines',
                    name=country_list[3]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_5'],
                    mode='lines',
                    name=country_list[4]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_6'],
                    mode='lines', 
                    name=country_list[5]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_7'],
                    mode='lines',
                    name=country_list[6]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_8'],
                    mode='lines',
                    name=country_list[7]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_9'],
                    mode='lines', 
                    name=country_list[8]))
fig_rolling7.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_10'],
                    mode='lines',
                    name=country_list[9]))
fig_rolling7.update_layout(title = 'Test conducted per confirmed case (7 day rolling average)')
fig_rolling7.update_xaxes(title= '------>Timeline' ,showline=False)
fig_rolling7.update_yaxes(title= '------>Tests / Confirmed Case', showline=False)
fig_rolling7.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=5
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=65, b=0),
        height=350)

#state wise

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
total = pd.DataFrame(statewise[statewise['State'] == 'Total'])
total = total[['Recovered', 'Deaths', 'Active']]
total = total.transpose()
total = total.reset_index()
total.rename(columns = {'index' : 'Property', 0 : 'Numbers'}, inplace = True)
statewise=statewise.drop([0])
df_confirmed1 = statewise[['State', 'Confirmed']]
df_recovered1 = statewise[['State', 'Recovered']]
df_death1 = statewise[['State', 'Deaths']]
df_active1 = statewise[['State', 'Active']]

statewise_5k = statewise[statewise['Confirmed'] >= 5000]
statewise_5k = statewise_5k[statewise_5k['State'] != 'State Unassigned']
statewise_5k = statewise_5k[statewise_5k['State'] != 'Total']
statewise_5k.sort_values(by=['Confirmed'], inplace=True,ascending=False)

fig_statewise = go.Figure(data=[
    go.Bar(name='Confirmed', x=statewise_5k['State'], y=statewise_5k['Confirmed'], marker_color = 'Blue'),
    go.Bar(name='Recovered', x=statewise_5k['State'], y=statewise_5k['Recovered'], marker_color = 'Green'),
    go.Bar(name='Deceased', x=statewise_5k['State'], y=statewise_5k['Deaths'], marker_color = 'Red')
])
# Change the bar mode
fig_statewise.update_layout(barmode='group')
fig_statewise.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=10, b=30),
        height=400)


net_updated = statewise[['State', 'Confirmed', 'Active', 'Recovered', 'Deaths']]
net_updated['Mortaliy Rate'] = net_updated['Deaths']/net_updated['Confirmed']
net_updated['Recovery Rate'] = net_updated['Recovered']/net_updated['Confirmed']
net_updated = net_updated.fillna(0)
net_updated.style.background_gradient(cmap = 'Reds')        
        
beds = pd.read_json('https://api.rootnet.in/covid19-in/hospitals/beds.json')
beds = pd.DataFrame(beds['data']['regional'])
statewise_beds = beds[['state', 'totalBeds']]
statewise_beds = statewise_beds[statewise_beds['state'] != 'INDIA']
statewise_beds = statewise_beds.sort_values(by = 'totalBeds', ascending = False)
fig_state_bed = px.bar(statewise_beds, 
            x = 'totalBeds',
            y = 'state', 
            orientation = 'h',
            color = 'state')
fig_state_bed.update_layout(
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=5
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
    )



urban_rural_beds = beds[['state', 'urbanHospitals','ruralHospitals']]
urban_rural_beds = urban_rural_beds[urban_rural_beds['state'] != 'INDIA']
urban_rural_beds.style.background_gradient(cmap='Greys')
fig_urban_rural = go.Figure(data=[
    go.Bar(name='Urban Hospitals', x=urban_rural_beds['state'], y=urban_rural_beds['urbanHospitals']),
    go.Bar(name='Rural Hospitals', x=urban_rural_beds['state'], y=urban_rural_beds['ruralHospitals'])
])
# Change the bar mode
fig_urban_rural.update_layout(barmode='group')
fig_urban_rural.update_layout(
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=5, b=30),
        height=400
    )

fig_urban_rural.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
fig_urban_rural.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
fig_urban_rural.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')

urban_rural_beds = beds[['state', 'urbanBeds','ruralBeds']]
urban_rural_beds = urban_rural_beds[urban_rural_beds['state'] != 'INDIA']
urban_rural_beds.style.background_gradient(cmap='Blues')
fig_urban_rural_beds = go.Figure(data=[
    go.Bar(name='Urban Beds', x=urban_rural_beds['state'], y=urban_rural_beds['urbanBeds'],marker_color = 'Wheat'),
    go.Bar(name='Rural Beds', x=urban_rural_beds['state'], y=urban_rural_beds['ruralBeds'],marker_color = 'Green')
])
fig_urban_rural_beds.update_layout(barmode='group')
fig_urban_rural_beds.update_layout(
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'], 
        margin=dict(l=0, r=0, t=10),
        height=400
    )

url = 'https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/locations.csv'
vaccine = pd.read_csv(url)
df2 = vaccine[['location','source_name','last_observation_date','vaccines']]
list1=df2.vaccines.unique()
list2=[]
for x in list1:
    if ',' in x:
        x = x.split(", ")
        list2.extend(x)
    else:
        list2.append(x)
old_list=set(list2)
new_list=list(old_list)
new_list
vaccine0=[]
len(vaccine0)
url = 'https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/vaccinations.csv'
vaccination = pd.read_csv(url)
df=vaccination
country_list=df.location.unique()
country_list_len=len(country_list)
united=df.loc[df['location'] == 'United States']
country=[]
for i in range(92):
    country_list[i]=df.loc[df['location'] == country_list[i]]
country_wise = df[df['location'] != 'world']
world = country_wise.sort_values(by = 'date')
world.rename(columns = {'total_vaccinations_per_hundred':'Vaccination/100'}, inplace = True)
fig_vcc_world_data = px.choropleth(world, locations="iso_code",
                    color="Vaccination/100",
                    hover_name="location",
                    animation_frame="date",
                    title = "Cumulative COVID-19 vaccination doses administered",
                   color_continuous_scale=px.colors.sequential.RdBu)
fig_vcc_world_data["layout"].pop("updatemenus")
fig_vcc_world_data.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'], 
        margin=dict(l=0, r=0),
        height=500,
    )

world['daily_vaccinations_per_million']=world['daily_vaccinations_per_million']/10000
world['daily_vaccinations_per_million']=world['daily_vaccinations_per_million'].round(2)
world.rename(columns = {'daily_vaccinations_per_million':'Vaccination/Million'}, inplace = True)

fig_vcc_per_mil = px.choropleth(world, locations="iso_code",
                    color="Vaccination/Million",
                    hover_name="location",
                    animation_frame="date",
                    title = "Daily COVID-19 vaccine doses administered per Million",
                   color_continuous_scale=px.colors.sequential.Aggrnyl)
fig_vcc_per_mil["layout"].pop("updatemenus")
fig_vcc_per_mil.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'], 
        margin=dict(l=0, r=0),
        height=500,
    )

for i in range(1,92):
    country_list[i]=country_list[i].fillna(0)
    country_list[i]=country_list[i].replace(to_replace=0, method='ffill')
frames = [country_list[0], country_list[1],country_list[2],country_list[3],country_list[4],country_list[5],country_list[6],country_list[7],country_list[8],country_list[9],country_list[10],country_list[11],country_list[12],country_list[13],country_list[14],country_list[15],country_list[16],country_list[17],country_list[18],country_list[19],country_list[20],country_list[21],country_list[22],country_list[23],country_list[24],country_list[25],country_list[26],country_list[27],country_list[28],country_list[29],country_list[30],country_list[31],country_list[32],country_list[33],country_list[34],country_list[35],country_list[36],country_list[37],country_list[38],country_list[39],country_list[40],country_list[41],country_list[42],country_list[43],country_list[44],country_list[45],country_list[46],country_list[47],country_list[48],country_list[49],country_list[50],country_list[51],country_list[52],country_list[53],country_list[54],country_list[55],country_list[56],country_list[57],country_list[58],country_list[59],country_list[60],country_list[61],country_list[62],country_list[63],country_list[64],country_list[65],country_list[66],country_list[67],country_list[68],country_list[69],country_list[70],country_list[71],country_list[72],country_list[73],country_list[74],country_list[75],country_list[76],country_list[77],country_list[78],country_list[79],country_list[80],country_list[81],country_list[82],country_list[83],country_list[84],country_list[85],country_list[86],country_list[87],country_list[88],country_list[89],country_list[90],country_list[91]]
result = pd.concat(frames)
world = result.sort_values(by = 'date')

fig_total_vacc = px.choropleth(world, locations="iso_code",
                    color="total_vaccinations",
                    hover_name="location",
                    animation_frame="date",
                    title = "COVID-19 vaccine doses administered",
                   color_continuous_scale=px.colors.sequential.RdBu)
fig_total_vacc["layout"].pop("updatemenus")
fig_total_vacc.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'], 
        margin=dict(l=0, r=0),
        height=500,
    )
result['people_vaccinated_per_hundred']=result['people_vaccinated_per_hundred'].round(1)
world1 = result.sort_values(by = 'date')
world1.rename(columns = {'people_vaccinated_per_hundred':'People Vaccinated/100'}, inplace = True)
fig_vacc_per_100 = px.choropleth(world1, locations="iso_code",
                    color="People Vaccinated/100",
                    hover_name="location",
                    animation_frame="date",
                    title = "Share of people who received at least one dose of COVID-19 vaccine",
                   color_continuous_scale=px.colors.sequential.Emrld)
fig_vacc_per_100["layout"].pop("updatemenus")
fig_vacc_per_100.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0),
        height=500,
    )

result['people_fully_vaccinated_per_hundred']=result['people_fully_vaccinated_per_hundred'].round(1)
world2 = result.sort_values(by = 'date')
resultt=result
resultt=resultt.set_index('location')
world3 = result.sort_values(by = 'date')
dfff = world3.set_index(['date'])
dfff=dfff.loc['2021-01-03':]
dfff=dfff.reset_index()
dfff.rename(columns = {'people_fully_vaccinated_per_hundred':'Fully Vaccinated/100'}, inplace = True)
fig_full_vacc = px.choropleth(dfff, locations="iso_code",
                    color="Fully Vaccinated/100",
                    hover_name="location",
                    animation_frame="date",
                    title = "Share of the population fully vaccinated against COVID-19",
                   color_continuous_scale=px.colors.sequential.Greens)
fig_full_vacc["layout"].pop("updatemenus")
fig_full_vacc.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0),
        height=500,
    )
dff=vaccination
world5 = dff.sort_values(by = 'date')
world5 = world5.set_index(['date'])
world5=world5.loc['2021-01-03':]
world5=world5.reset_index()
fig_full_vaccinated = px.choropleth(world5, locations="iso_code",
                    color="people_fully_vaccinated",
                    hover_name="location",
                    animation_frame="date",
                    title = "Number of people fully vaccinated against COVID-19",
                   color_continuous_scale=px.colors.sequential.Greens)
fig_full_vaccinated["layout"].pop("updatemenus")
fig_full_vaccinated.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0),
        height=500,
    )

url = 'http://api.covid19india.org/csv/latest/cowin_vaccine_data_statewise.csv'
vaccine_by_state = pd.read_csv(url)
vaccine_by_state.rename(columns = {'Updated On':'date'}, inplace = True)
vaccine_by_state['date'] = pd.to_datetime(vaccine_by_state['date'])
vaccine_by_state = vaccine_by_state.set_index(['date'])
today=vaccine_by_state.tail(1).index.item()
vaccine_by_state=vaccine_by_state.reset_index()
todayy = date.today()
todayy = str(todayy)
vaccine_by_state['date'] = vaccine_by_state['date'].apply(str)
vaccine_by_state = vaccine_by_state.sort_values(by = 'date')
vaccine_by_state = vaccine_by_state.set_index(['date'])
vaccine_by_state=vaccine_by_state.loc['2021-01-16':todayy]
vaccine_by_state=vaccine_by_state.reset_index()
fig_vacc_total_dose = px.choropleth(
    vaccine_by_state,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='State',
    hover_name='State',
    hover_data = ['First Dose Administered', 'Second Dose Administered','Male(Individuals Vaccinated)','Female(Individuals Vaccinated)','Transgender(Individuals Vaccinated)'],
    animation_frame="date",
    title = "Vaccination Gender Distrubution",
    color='Total Doses Administered',
    color_continuous_scale='oranges'
)
fig_vacc_total_dose.update_geos(fitbounds="locations", visible=False)

fig_vacc_total_dose["layout"].pop("updatemenus")
fig_vacc_total_dose.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0),
        height=500,
    )

fig_vacc_reg = px.choropleth(
    vaccine_by_state,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='State',
    hover_name='State',
    hover_data = ['18-30 years(Age)','30-45 years(Age)','45-60 years(Age)','60+ years(Age)', 'Total Individuals Vaccinated'],
    animation_frame="date",
    title = "Age Wise Covid Vaccination",
    color='Total Doses Administered',
    color_continuous_scale='oranges'
)
fig_vacc_reg.update_geos(fitbounds="locations", visible=False)

fig_vacc_reg["layout"].pop("updatemenus")
fig_vacc_reg.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0),
        height=500,
    )

fig_covishield = px.choropleth(
    vaccine_by_state,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='State',
    hover_name='State',
    hover_data = ['Total CoviShield Administered'],
    animation_frame="date",
    color='Total CoviShield Administered',
    title = "CovidShield",
    color_continuous_scale='earth'
)
fig_covishield.update_geos(fitbounds="locations", visible=False)
fig_covishield["layout"].pop("updatemenus")
fig_covishield.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0),
        height=500,
    )

fig_covaxin = px.choropleth(
    vaccine_by_state,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='State',
    hover_name='State',
    hover_data = ['Total Covaxin Administered'],
    animation_frame="date",
    color='Total Covaxin Administered',
    title = "Covaxin",
    color_continuous_scale='earth'
)
fig_covaxin.update_geos(fitbounds="locations", visible=False)

fig_covaxin["layout"].pop("updatemenus")
fig_covaxin.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0),
        height=500,
    )

india_vaccine_data=vaccine_by_state.loc[vaccine_by_state['State'] == 'India']
vaccine_by_gender=india_vaccine_data[['date','State','Male(Individuals Vaccinated)','Female(Individuals Vaccinated)','Transgender(Individuals Vaccinated)']]


fig_log_one = go.Figure(go.Line(x = vaccine_by_gender['date'], 
                            y = (vaccine_by_gender['Male(Individuals Vaccinated)']),
                            name = 'Male(Individuals Vaccinated)', 
                            mode = 'lines'))
fig_log_one.add_trace(go.Line(x = vaccine_by_gender['date'],
                          y = (vaccine_by_gender['Female(Individuals Vaccinated)']), 
                          name = 'Female(Individuals Vaccinated', 
                          mode = 'lines'))
fig_log_one.add_trace(go.Line(x = vaccine_by_gender['date'],
                          y = (vaccine_by_gender['Transgender(Individuals Vaccinated)']), 
                          name = 'Transgender(Individuals Vaccinated)', 
                          mode = 'lines'))
fig_log_one.update_layout(title = 'INDIA')
fig_log_one.update_xaxes(title= '------>Timeline' ,showline=False)
fig_log_one.update_layout(
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor='rgba(44,53,57, 0.8)',
        plot_bgcolor='rgba(44,53,57, 0.8)', 
        margin=dict(l=0, r=0),
        height=500,
    )
vaccine_by_company=india_vaccine_data[['date','State','Total Covaxin Administered','Total CoviShield Administered']]
fig_log_two = go.Figure(go.Line(x = vaccine_by_gender['date'], 
                            y = (vaccine_by_company['Total Covaxin Administered']),
                            name = 'Total Covaxin Administered', 
                            mode = 'lines'))
fig_log_two.add_trace(go.Line(x = vaccine_by_gender['date'],
                          y = (vaccine_by_company['Total CoviShield Administered']), 
                          name = 'Total CoviShield Administered', 
                          mode = 'lines'))
fig_log_two.update_layout(title = 'INDIA')
fig_log_two.update_xaxes(title= '------>Timeline' ,showline=False)
fig_log_two.update_layout(
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor='rgba(44,53,57, 0.8)',
        plot_bgcolor='rgba(44,53,57, 0.8)', 
        margin=dict(l=0, r=0),
        height=500,
    )
#fig_log_one.update_yaxes(title= '------> increment in cases', showline=False)

vaccine_session_doses=india_vaccine_data[['date','State','Total Sessions Conducted','First Dose Administered','Second Dose Administered']]
fig_log_three = go.Figure(go.Line(x = vaccine_session_doses['date'], 
                            y = (vaccine_session_doses['Total Sessions Conducted']),
                            name = 'Total Sessions Conducted', 
                            mode = 'lines'))
fig_log_three.add_trace(go.Line(x = vaccine_session_doses['date'],
                          y = (vaccine_session_doses['First Dose Administered']), 
                          name = 'First Dose Administered', 
                          mode = 'lines'))
fig_log_three.add_trace(go.Line(x = vaccine_session_doses['date'],
                          y = (vaccine_session_doses['Second Dose Administered']), 
                          name = 'Second Dose Administered', 
                          mode = 'lines'))
fig_log_three.update_layout(title = 'INDIA')
fig_log_three.update_xaxes(title= '------>Timeline' ,showline=False)
fig_log_three.update_layout(
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor='rgba(44,53,57, 0.8)',
        plot_bgcolor='rgba(44,53,57, 0.8)', 
        margin=dict(l=0, r=0),
        height=500,
    )
vaccine_by_doses=india_vaccine_data[['date','State','Total Individuals Vaccinated','Total Doses Administered']]
fig_log_four = go.Figure(go.Line(x = vaccine_by_doses['date'],
                          y = (vaccine_by_doses['Total Individuals Vaccinated']), 
                          name = 'Total Individuals Vaccinated', 
                          mode = 'lines'))
fig_log_four.add_trace(go.Line(x = vaccine_by_doses['date'],
                          y = (vaccine_by_doses['Total Doses Administered']), 
                          name = 'Total Doses Administered', 
                          mode = 'lines'))
fig_log_four.update_layout(title = 'INDIA')
fig_log_four.update_xaxes(title= '------>Timeline' ,showline=False)
fig_log_four.update_layout(
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor='rgba(44,53,57, 0.8)',
        plot_bgcolor='rgba(44,53,57, 0.8)', 
        margin=dict(l=0, r=0),
        height=500,
    )
url = 'https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/locations.csv'
vaccine_plotting = pd.read_csv(url)
fig_vacc_world_data = px.choropleth(vaccine_plotting, locations="iso_code",
                    #color="vaccines",
                    hover_name="location",
                    hover_data=['vaccines'],
                    #animation_frame="date",
                   color_continuous_scale="Viridis")
fig_vacc_world_data["layout"].pop("updatemenus")
fig_vacc_world_data.update_layout(
        coloraxis_showscale=False,
        geo=dict(bgcolor= 'rgba(44,53,57, 0.8)'),
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=0,
        ),
        paper_bgcolor='rgba(44,53,57, 0.8)',
        plot_bgcolor='rgba(44,53,57, 0.8)', 
        margin=dict(l=0, r=0),
        height=500,
    )

Arima_Active = base64.b64encode(open('image/Arima_Active.png', 'rb').read())
Arima_Confirmed = base64.b64encode(open('image/Arima_Confirmed.png', 'rb').read())
Arima_Deaceased = base64.b64encode(open('image/Arima_Deaceased.png', 'rb').read())
Arima_Recoverd = base64.b64encode(open('image/Arima_Recovered.png', 'rb').read())

Prophet_Active = base64.b64encode(open('image/Prophet_Active.png', 'rb').read())
Prophet_Confirmed = base64.b64encode(open('image/Prophet_Confirmed.png', 'rb').read())
Prophet_Deaceased = base64.b64encode(open('image/Prophet_Death.png', 'rb').read())
Prophet_Recoverd = base64.b64encode(open('image/Prophet_Recovered.png', 'rb').read())

Arima_Vacc = base64.b64encode(open('image/Arima_Vacc.png', 'rb').read())
Prophet_Vacc = base64.b64encode(open('image/Prophet_Vacc.png', 'rb').read())


# Bootstrap Grid Layout


app.layout = html.Div(
    html.Div([
        # Header display
        html.Div(
            [
                html.H1(children='COVID-19 Interactive Tracker',
                        style={
                            'textAlign': 'left',
                            'color': colors['text'],
                            'backgroundColor': colors['background'],
                        },
                        className='ten columns',
                        ),

                html.Div([
                    html.Button(html.I(className="fa fa-info-circle"),
                        id='info-button',
                        style={
                             'color': colors['text'],
                             'fontSize':'36px'

                         },)

                ],className='two columns',),

                # Preload Modal windows and set "display": "none" to hide it first
                html.Div([  # modal div
                    html.Div([  # content div

                        dcc.Markdown('''
                            ##### Dataset provided by Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE):
                            https://systems.jhu.edu/
                           
                            Data Sources:
                            * World Health Organization (WHO): https://www.who.int/
                            * DXY.cn. Pneumonia. 2020. http://3g.dxy.cn/newh5/view/pneumonia.
                            * BNO News: https://bnonews.com/index.php/2020/02/the-latest-coronavirus-cases/
                            * National Health Commission of the People’s Republic of China (NHC):
                            http://www.nhc.gov.cn/xcs/yqtb/list_gzbd.shtml
                            * China CDC (CCDC): http://weekly.chinacdc.cn/news/TrackingtheEpidemic.htm
                            * Hong Kong Department of Health: https://www.chp.gov.hk/en/features/102465.html
                            * Macau Government: https://www.ssm.gov.mo/portal/
                            * Taiwan CDC: https://sites.google.com/cdc.gov.tw/2019ncov/taiwan?authuser=0
                            * US CDC: https://www.cdc.gov/coronavirus/2019-ncov/index.html
                            * Government of Canada: https://www.canada.ca/en/public-health/services/diseases/coronavirus.html
                            * Australia Government Department of Health: https://www.health.gov.au/news/coronavirus-update-at-a-glance
                            * European Centre for Disease Prevention and Control (ECDC): https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases 
                            * Ministry of Health Singapore (MOH): https://www.moh.gov.sg/covid-19
                            * Italy Ministry of Health: http://www.salute.gov.it/nuovocoronavirus
                            * 1Point3Arces: https://coronavirus.1point3acres.com/en
                            * WorldoMeters: https://www.worldometers.info/coronavirus/
                            '''),
                        html.Hr(),
                        html.Button('Close', id='modal-close-button',
                        style={
                             'color': colors['text'],
                         },)
                    ],
                        style={
                            'fontSize': 10,
                            'lineHeight': 0.9,
                        },
                        className='modal-content',
                    ),
                ],
                    id='modal',
                    className='modal',
                    style={"display": "none"},
                ),

                html.Div([html.Span('Dashboard: COVID-19 outbreak. (Updated once a day, based on consolidated last day total) Last Updated: ',
                             style={'color': colors['text'],
                             }),
                        html.Span(datatime_convert(df_confirmed.columns[-1],1) + '  00:01 (UTC).',
                             style={'color': colors['confirmed_text'],
                             'fontWeight': 'bold',
                             }),

                         ],className='twelve columns'
                         ),
                html.Div([html.Span('Outbreak since 22-Jan-2020: ',
                             style={'color': colors['text'],
                             }),
                        html.Span(str(return_outbreakdays(datatime_convert(df_confirmed.columns[-1],1))) + '  days.',
                             style={'color': colors['confirmed_text'],
                             'fontWeight': 'bold',
                             }),

                         ],
                         className='twelve columns'
                         ),
                html.Div(children='Best viewed on Desktop. Refresh browser for latest update.',
                         style={
                             'textAlign': 'left',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                         },
                         className='twelve columns'
                         )
            ], className="row"
        ),
        html.Div([
            
            html.Div([

                html.Div([

                    html.Div([

                        html.Div([
                            html.Span(
                            ['Home'],
                            className="three.columns"
                            ),
                            html.Span(
                            ['Prediction'],
                            className="three.columns"
                            ),
                            html.Span(
                            ['Vaccsine'],
                            className="three.columns"
                            ),
                        ],
                        className="nav navbar-nav"
                        ),

                    ],
                    className="collapse navbar-collapse",
                    id="navbar-primary-collapse"
                    ),

                ],
                className="container-fluid"
                ),

             ],
            className="navbar navbar-default",
            id="navbar-primary"
            ),

        ],className="row"
        ),

        # Top column display of confirmed, death and recovered total numbers
        html.Div([
            html.H4(children='Global Covid-19 cases',
                         style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],

                         },
                         className='twelve columns'
                         ),
            html.Div([
                html.H4(children='Total Cases: ',
                       style={
                           'textAlign': 'center',
                           'color': colors['confirmed_text'],
                       }
                       ),
                html.P(f"{df_confirmed_total[-1]:,d}",
                       style={
                    'textAlign': 'center',
                    'color': colors['confirmed_text'],
                    'fontSize': 30,
                }
                ),
                html.P('Past 24hrs increase: +' + f"{df_confirmed_total[-1] - df_confirmed_total[-2]:,d}"
                       + ' (' + str(round(((df_confirmed_total[-1] - df_confirmed_total[-2])/df_confirmed_total[-1])*100, 2)) + '%)',
                       style={
                    'textAlign': 'center',
                    'color': colors['confirmed_text'],
                }
                ),
            ],
                style=divBorderStyle,
                className='four columns',
            ),
            html.Div([
                html.H4(children='Total Deceased: ',
                       style={
                           'textAlign': 'center',
                           'color': colors['deaths_text'],
                       }
                       ),
                html.P(f"{df_deaths_total[-1]:,d}",
                       style={
                    'textAlign': 'center',
                    'color': colors['deaths_text'],
                    'fontSize': 30,
                }
                ),
                html.P('Mortality Rate: ' + str(round(df_deaths_total[-1]/df_confirmed_total[-1] * 100, 3)) + '%',
                       style={
                    'textAlign': 'center',
                    'color': colors['deaths_text'],
                }
                ),
            ],
                style=divBorderStyle,
                className='four columns'),
            html.Div([
                html.H4(children='Total Recovered: ',
                       style={
                           'textAlign': 'center',
                           'color': colors['recovered_text'],
                       }
                       ),
                html.P(f"{df_recovered_total[-1]:,d}",
                       style={
                    'textAlign': 'center',
                    'color': colors['recovered_text'],
                    'fontSize': 30,
                }
                ),
                html.P('Recovery Rate: ' + str(round(df_recovered_total[-1]/df_confirmed_total[-1] * 100, 3)) + '%',
                       style={
                    'textAlign': 'center',
                    'color': colors['recovered_text'],
                }
                ),
            ],
                style=divBorderStyle,
                className='four columns'),
        ], className='row'
        ),

        html.Div([
            html.Div([

                    html.P([html.Span('Countries with highest cases: ',
                             ),
                    html.Br(),
                    html.Span(' + past 24hrs',
                             style={'color': colors['confirmed_text'],
                             'fontWeight': 'bold','fontSize': 14,})
                    ],
                    style={
                        'textAlign': 'center',
                        'color': 'rgb(200,200,200)',
                        'fontsize':12,
                        'backgroundColor':'#3B5998',
                        'borderRadius': '12px',
                        'fontSize': 17,
                        }       
                ),
                html.P(confirm_cases),
            ],
                className="three columns",
            ),

            html.Div([
                    html.P([html.Span('Single day highest cases: ',
                             ),
                    html.Br(),
                    html.Span(' + past 24hrs',
                             style={'color': colors['confirmed_text'],
                             'fontWeight': 'bold','fontSize': 14,})
                    ],
                    style={
                        'textAlign': 'center',
                        'color': 'rgb(200,200,200)',
                        'fontsize':12,
                        'backgroundColor':'#3B5998',
                        'borderRadius': '12px',
                        'fontSize': 17,
                        }       
                ),

                html.P(confirm_cases_24hrs),
            ],
                className="three columns",
            ),
             html.Div([
                    html.P([html.Span('Countries with highest mortality: ',
                             ),
                    html.Br(),
                    html.Span(' + past 24hrs (Mortality Rate)',
                             style={'color': '#f2786f',
                             'fontWeight': 'bold','fontSize': 14,})
                    ],
                    style={
                        'textAlign': 'center',
                        'color': 'rgb(200,200,200)',
                        'fontsize':12,
                        'backgroundColor':'#ab2c1a',
                        'borderRadius': '12px',
                        'fontSize': 17,
                        }       
                ),

                html.P(deaths_cases),
            ],
                className="three columns",
            ),
            html.Div([

                    html.P([html.Span('Single day highest mortality: ',
                             ),
                    html.Br(),
                    html.Span(' + past 24hrs (Mortality Rate)',
                             style={'color': '#f2786f',
                             'fontWeight': 'bold','fontSize': 14,})
                    ],
                    style={
                        'textAlign': 'center',
                        'color': 'rgb(200,200,200)',
                        'fontsize':12,
                        'backgroundColor':'#ab2c1a',
                        'borderRadius': '12px',
                        'fontSize': 17,
                        }       
                ),

                html.P(deaths_cases_24hrs),
            ],
                className="three columns",
            ),
        ], className="row",
            style={
            'textAlign': 'left',
            'color': colors['text'],
            'backgroundColor': colors['background'],
                'padding': 20
        },
        ),

        # Graph of total confirmed, recovered and deaths
        html.Div(
            [
                
                html.Div([
                    dcc.RadioItems(
                        id='graph-type',
                        options=[{'label': i, 'value': i}
                                 for i in ['Total Cases', 'Daily Cases']],
                        value='Total Cases',
                        labelStyle={'display': 'inline-block'},
                        style={
                            'fontSize': 20,
                         },
                        
                    )
                ],className='six columns'
                ),
                                html.Div([
                    dcc.RadioItems(
                        id='graph-high10-type',
                        options=[{'label': i, 'value': i}
                                 for i in ['Confirmed Cases', 'Deceased Cases']],
                        value='Confirmed Cases',
                        labelStyle={'display': 'inline-block'},
                        style={
                            'fontSize': 20,
                         },
                        
                    )
                ],className='six columns'
                ),
                html.Div([
                    dcc.Graph(
                        id='global-graph',

                    )
                ], className='six columns'
                ),
                html.Div([
                    dcc.Graph(
                        id='high10-graph',

                    )
                ], className='six columns'
                ),

            ], className="row",
            style={
                'textAlign': 'left',
                'color': colors['text'],
                'backgroundColor': colors['background'],
            },
        ),

        # Highest 5 Countries Display
        # 1x4 grid
       

        html.Div([
            html.Br(),
            html.H4(children='Global Outbreak Map - Select row from table to locate in map',
                     style={
                         'textAlign': 'center',
                         'color': colors['text'],
                         'backgroundColor': colors['background'],
                     },
                     className='twelve columns'
                     ),
        ], className='row'
        ),

        #Map, Table
        html.Div(
            [
                html.Br(),
                html.Div(
                    [
                        dcc.Graph(id='map-graph',
                                  )
                    ], className="six columns"
                ),
                html.Div(
                    [
                        dt.DataTable(
                            data=map_data.to_dict('records'),
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": True} for i in ['Province/State', 'Country/Region', 'Confirmed',
                                                                                                       'Deaths', 'Recovered']
                            ],
                            fixed_rows={'headers': True, 'data': 0},
                            style_header={
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'fontWeight': 'bold'
                            },
                            style_cell={
                                'backgroundColor': 'rgb(100, 100, 100)',
                                'color': colors['text'],
                                'maxWidth': 0,
                                'fontSize':14,
                            },
                            style_table={
                                'maxHeight': '350px',
                                'overflowY': 'auto'
                            },
                            style_data={
                                'whiteSpace': 'normal',
                                'height': 'auto',

                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'even'},
                                    'backgroundColor': 'rgb(60, 60, 60)',
                                },
                                {
                                    'if': {'column_id' : 'Confirmed'},
                                    'color':colors['confirmed_text'],
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {'column_id' : 'Deaths'},
                                    'color':colors['deaths_text'],
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {'column_id' : 'Recovered'},
                                    'color':colors['recovered_text'],
                                    'fontWeight': 'bold'
                                },
                                ],
                            style_cell_conditional=[
                                {'if': {'column_id': 'Province/State'},
                                 'width': '26%'},
                                {'if': {'column_id': 'Country/Region'},
                                 'width': '26%'},
                                {'if': {'column_id': 'Confirmed'},
                                 'width': '16%'},
                                {'if': {'column_id': 'Deaths'},
                                 'width': '11%'},
                                {'if': {'column_id': 'Recovered'},
                                 'width': '16%'},
                            ],

                            editable=False,
                            filter_action="native",
                            sort_action="native",
                            sort_mode="single",
                            row_selectable="single",
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_current=0,
                            page_size=1000,
                            id='datatable'
                        ),
                    ],
                    className="six columns"
                ),

            ], className="row",
        ),

        #Single country line graph, single country bar graph
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='line-graph',
                                  )
                    ], className="six columns"
                ),
                html.Div(
                    [
                        dcc.Graph(id='bar-graph',
                                  )
                    ], className="six columns"
                ),

            ], className="row",
            style={
                'textAlign': 'left',
                'color': colors['text'],
                'backgroundColor': colors['background'],
            },
        ),

        html.Div([
            html.Div([html.P()],className='six columns'), #leave a gap of 6 columns
            html.Div([
                dcc.RadioItems(
                    id='graph-line',
                    options=[{'label': i, 'value': i}
                             for i in ['Bar Chart', 'Line Chart']],
                    value='Line Chart',
                    labelStyle={'display': 'inline-block'},
                    style={
                        'fontSize': 20,
                        'textAlign': 'left',
                    },

                )
            ],className="six columns"),  
        ],className="row"),

        html.Div([
            html.Br(),
            html.H4(children='Bed Availability',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),

            html.Div([
                    dcc.Graph(figure=fig_bed_per_1000)

                ], className='six columns'
                ),

            html.Div([
                    dcc.Graph(figure=fig_rolling7)
                ], className='six columns'
                ),

        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.H4(children='Covid 19 Cases Statewise',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
            html.Div([
                    dcc.Graph(figure=fig_statewise)
                ], className='twelve columns'
                ),

        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.H4(children='No of Hospitals in each State',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
            html.Div([
                    dcc.Graph(figure=fig_urban_rural)
                ], className='twelve columns'
                ),

        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.H4(children='Hospital Beds In Each State',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
            html.Div([
                    dcc.Graph(figure=fig_state_bed)
                ], className='twelve columns'
                ),

        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.H4(children='Urban and Rural Hospital Beds In Each State',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
            html.Div([
                    dcc.Graph(figure=fig_urban_rural_beds)
                ], className='twelve columns'
                ),

        ], className='row'
        ),
        html.Div([
            html.Br(),
            html.H4(children='Covid-19 Vaccination used by Dfferent Countries',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
            html.Div([
                    dcc.Graph(figure=fig_vacc_world_data)
                ], className='twelve columns'
                ),

        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.H4(children='Covid-19 vaccination doses administered ',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
            html.Div([
                    dcc.Graph(figure=fig_vcc_world_data)
                ], className='six columns'
                ),
            html.Div([
                    dcc.Graph(figure=fig_vcc_per_mil)
                ], className='six columns'
                ),

        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.Div([
                    dcc.Graph(figure=fig_total_vacc)
                ], className='six columns'
                ),
            html.Div([
                    dcc.Graph(figure=fig_vacc_per_100)
                ], className='six columns'
                ),

        ], className='row'
        ), 
        html.Div([
            html.Br(),
            html.Div([
                    dcc.Graph(figure=fig_full_vacc)
                ], className='six columns'
                ),
            html.Div([
                    dcc.Graph(figure=fig_full_vaccinated)
                ], className='six columns'
                ),

        ], className='row'
        ), 

        html.Div([
            html.Br(),
            html.H4(children='Covid-19 vaccination doses administered In India',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
            html.Div([
                    dcc.Graph(figure=fig_log_one)
                ], className='twelve columns'
                ),

        ], className='row'
        ),
        html.Div([
            html.Br(),
            html.Div([
                    dcc.Graph(figure=fig_log_three)
                ], className='six columns'
                ),
            html.Div([
                    dcc.Graph(figure=fig_log_four)
                ], className='six columns'
                ),

        ], className='row'
        ),
        html.Div([
            html.Br(),
            html.Div([
                    dcc.Graph(figure=fig_vacc_total_dose)
                ], className='six columns'
                ),
            html.Div([
                    dcc.Graph(figure=fig_vacc_reg)
                ], className='six columns'
                ),

        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.H4(children='CoviShield Vs Covaxin',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),

            html.Div([
                    dcc.Graph(figure=fig_log_two)
                ], className='twelve columns'
                ),
            

        ], className='row'
        ),
        html.Div([
            html.Br(),
            html.Div([
                    dcc.Graph(figure=fig_covishield)
                ], className='six columns'
                ),
            html.Div([
                    dcc.Graph(figure=fig_covaxin)
                ], className='six columns'
                ),

        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.H4(children='Covid-19 Cases Rise and Fall Prediction Using ARIMA and Prophet Model',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
        ], className='row'
        ),

        html.Div([
            html.Br(),

            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Arima_Confirmed.decode())),
                ], className='six columns'
                ),
            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Arima_Active.decode())),
                ], className='six columns'
                ),
        ], className='row'
        ),

        html.Div([
            html.Br(),
            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Prophet_Confirmed.decode())),
                ], className='twelve columns'
                ),
            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Prophet_Active.decode())),
                ], className='twelve columns'
                ),
        ], className='row'
        ),

        html.Div([
            html.Br(),

            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Arima_Deaceased.decode())),
                ], className='six columns'
                ),
            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Arima_Recoverd.decode())),
                ], className='six columns'
                ),
            
        ], className='row'
        ),

        html.Div([
            html.Br(),

            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Prophet_Deaceased.decode())),
                ], className='twelve columns'
                ),
            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Prophet_Recoverd.decode())),
                ], className='twelve columns'
                ),
        ], className='row'
        ),
        

        html.Div([
            html.Br(),
            html.H4(children='Vaccination Prediction',
                        style={
                             'textAlign': 'center',
                             'color': colors['text'],
                             'backgroundColor': colors['background'],
                        },
            className='twelve columns'
            ),
            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Arima_Vacc.decode())),
                ], className='twelve columns'
                ),
            
        ], className='row'
        ),
        html.Div([
            html.Br(),
            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(Prophet_Vacc.decode())),
                ], className='twelve columns'
                ),
            html.Br(),
            html.Br(),
        ], className='row'
        ),
    ],
    className='ten columns offset-by-one'
    ),
    style={
        'textAlign': 'left',
        'color': colors['text'],
        'backgroundColor': colors['background'],
    },
)


@app.callback(
    Output('global-graph', 'figure'),
    [Input('graph-type', 'value')])
def update_graph(graph_type):
    fig_global = draw_global_graph(df_confirmed_total,df_deaths_total,df_recovered_total,graph_type)
    return fig_global

@app.callback(
    Output('high10-graph', 'figure'),
    [Input('graph-high10-type', 'value')])
def update_graph_high10(graph_high10_type):
    fig_high10 = draw_highest_10(df_confirmed_t_stack, df_deaths_t_stack, graph_high10_type)
    return fig_high10

@app.callback(
    [Output('map-graph', 'figure'),
    Output('line-graph', 'figure'),
    Output('bar-graph', 'figure')],
    [Input('datatable', 'data'),
     Input('datatable', 'selected_rows'),
     Input('graph-line','value')])
def map_selection(data, selected_rows,graph_line):
    aux = pd.DataFrame(data)
    temp_df = aux.iloc[selected_rows, :]
    zoom = 1
    if len(selected_rows) == 0:
        fig1 = draw_singleCountry_Scatter(df_confirmed_t,df_deaths_t,df_recovered_t,0)
        fig2 = draw_singleCountry_Bar(df_confirmed_t,df_deaths_t,df_recovered_t,0,graph_line)
        return gen_map(aux,zoom,1.2833,103.8333), fig1,fig2
    else:
        fig1 = draw_singleCountry_Scatter(df_confirmed_t,df_deaths_t,df_recovered_t,selected_rows[0])
        fig2 = draw_singleCountry_Bar(df_confirmed_t,df_deaths_t,df_recovered_t,selected_rows[0],graph_line)
        zoom=4
        return gen_map(aux,zoom,temp_df['Lat'].iloc[0],temp_df['Long'].iloc[0]), fig1,fig2

# hide/show modal
@app.callback(Output('modal', 'style'),
              [Input('info-button', 'n_clicks')])
def show_modal(n):
    if n > 0:
        return {"display": "block"}
    return {"display": "none"}

# Close modal by resetting info_button click to 0
@app.callback(Output('info-button', 'n_clicks'),
              [Input('modal-close-button', 'n_clicks')])
def close_modal(n):
    return 0

if __name__ == '__main__':
    app.run_server(debug=True)