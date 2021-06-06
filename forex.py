
### Data Science Capestone Project for General Assembly by Samay Shah, June 2021.


import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from plotly import __version__
import cufflinks as cf
from fbprophet import Prophet

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px
import pickle
from datetime import datetime
import datetime

model_usd_daily = pickle.load(open('./Saved_Models_Pickle/model_usd_daily.pkl','rb'))
model_inr_daily = pickle.load(open('./Saved_Models_Pickle/model_inr_daily.pkl','rb'))
model_gbp_daily = pickle.load(open('./Saved_Models_Pickle/model_gbp_daily.pkl','rb'))

model_usd_monthly = pickle.load(open('./Saved_Models_Pickle/model_usd_monthly.pkl','rb'))
model_inr_monthly = pickle.load(open('./Saved_Models_Pickle/model_inr_monthly.pkl','rb'))
model_gbp_monthly = pickle.load(open('./Saved_Models_Pickle/model_gbp_monthly.pkl','rb'))


#---------------------------------#
#### Datasets from Mainfile to generate regular plots,


## Import data file
## Exchange Rate Information
data_inr = pd.read_csv("./Data/INR_CAD_hist_2004_28_05_2021.csv")
data_usd = pd.read_csv('./Data/USD_CAD_hist_2004_28_05_2021.csv')
data_gbp = pd.read_csv("./Data/GBP_CAD_hist_2004_28_05_2021.csv")
## Stock Market Information
ind_sensex = pd.read_csv('./Data/india_sensex_formodelling.csv')
usd_nifty = pd.read_csv('./Data/nifty_modelling.csv')
gbp_lse = pd.read_csv("./Data/LSE_formodelling.csv")
can_tse = pd.read_csv("./Data/tse_modelling.csv")
## Inflation Information
ind_inflation = pd.read_csv('./Data/indiainflation.csv')
usd_inflation = pd.read_csv('./Data/usinflation.csv')
gbp_inflation = pd.read_csv('./Data/uk_cpi.csv')
can_inflation = pd.read_csv('./Data/canadainflation.csv')
## GDP Information (additional feature for all)
can_gdp = pd.read_csv('./Data/canada_gdp2.csv')


### FOR DAILY PREDICTION
## Set Date Column as index for all

data_inr.Date = pd.to_datetime(data_inr.Date, dayfirst = True)
data_inr.set_index("Date", inplace=True)

data_usd.Date = pd.to_datetime(data_usd.Date, dayfirst = True)
data_usd.set_index("Date", inplace=True)

data_gbp.Date = pd.to_datetime(data_gbp.Date, dayfirst = True)
data_gbp.set_index("Date", inplace=True)

ind_sensex.Date = pd.to_datetime(ind_sensex.Date, dayfirst = True)
ind_sensex.set_index("Date", inplace=True)

usd_nifty.Date = pd.to_datetime(usd_nifty.Date, dayfirst = True)
usd_nifty.set_index("Date", inplace=True)

gbp_lse.Date = pd.to_datetime(gbp_lse.Date, dayfirst = True)
gbp_lse.set_index("Date", inplace=True)

can_tse.Date = pd.to_datetime(can_tse.Date, dayfirst = True)
can_tse.set_index("Date", inplace=True)

ind_inflation.Date = pd.to_datetime(ind_inflation.Date, dayfirst = True)
ind_inflation.set_index("Date", inplace=True)

usd_inflation.Date = pd.to_datetime(usd_inflation.Date, dayfirst = True)
usd_inflation.set_index("Date", inplace=True)

gbp_inflation.Date = pd.to_datetime(gbp_inflation.Date, dayfirst = True)
gbp_inflation.set_index("Date", inplace=True)

can_inflation.Date = pd.to_datetime(can_inflation.Date, dayfirst = True)
can_inflation.set_index("Date", inplace=True)

can_gdp.Date = pd.to_datetime(can_gdp.Date, dayfirst = True)
can_gdp.set_index("Date", inplace=True)

## Monthly/Quarterly Data -> Daily and fill missing values using forward fill imputation

data_usd_daily = data_usd.resample('B').mean().replace(0,np.nan)
data_usd_daily = data_usd_daily.fillna(method='ffill')

data_inr_daily = data_inr.resample('B').mean().replace(0,np.nan)
data_inr_daily = data_inr_daily.fillna(method='ffill')

data_gbp_daily = data_gbp.resample('B').mean().replace(0,np.nan)
data_gbp_daily = data_gbp_daily.fillna(method='ffill')

ind_sensex_daily = ind_sensex.resample('B').mean().replace(0,np.nan)
ind_sensex_daily = ind_sensex_daily.fillna(method='ffill')

usd_nifty_daily = usd_nifty.resample('B').mean().replace(0,np.nan)
usd_nifty_daily = usd_nifty_daily.fillna(method='ffill')

gbp_lse_daily = gbp_lse.resample('B').mean().replace(0,np.nan)
gbp_lse_daily = gbp_lse_daily.fillna(method='ffill')

can_tse_daily = can_tse.resample('B').mean().replace(0,np.nan)
can_tse_daily = can_tse_daily.fillna(method='ffill')

ind_inflation = ind_inflation.astype(float).replace(0,np.nan).replace('.',0)
ind_inflation_daily = ind_inflation.resample('B').mean()
ind_inflation_daily = ind_inflation_daily.fillna(method='ffill')

usd_inflation = usd_inflation.replace(0,np.nan).replace('.',0)
usd_inflation = usd_inflation.astype(float)
usd_inflation_daily = usd_inflation.resample('B').mean()
usd_inflation_daily = usd_inflation_daily.fillna(method='ffill')

gbp_inflation = gbp_inflation.astype(float).replace(0,np.nan).replace('.',0)
gbp_inflation_daily = gbp_inflation.resample('B').mean()
gbp_inflation_daily = gbp_inflation_daily.fillna(method='ffill')

can_inflation = can_inflation.astype(float).replace(0,np.nan).replace('.',0)
can_inflation_daily = can_inflation.resample('B').mean()
can_inflation_daily = can_inflation_daily.fillna(method='ffill')

can_gdp_daily = can_gdp.resample('B').mean().replace(0,np.nan)
can_gdp_daily = can_gdp_daily.fillna(method='ffill')

## Creating a dataframe for 'daily' data
daily_data = pd.concat([data_inr_daily, data_usd_daily, data_gbp_daily, ind_sensex_daily, usd_nifty_daily, gbp_lse_daily, can_tse_daily, ind_inflation_daily, usd_inflation_daily, gbp_inflation_daily, can_inflation_daily, can_gdp_daily], axis=1)
daily_data = daily_data.astype(float)
daily_data.drop(['Unnamed: 2'],axis=1,inplace=True)

### FOR MONTHLY Prediction
data_usd_monthly = data_usd.resample('M').mean().replace(0,np.nan)
data_usd_monthly = data_usd_monthly.fillna(method='ffill')

data_inr_monthly = data_inr.resample('M').mean().replace(0,np.nan)
data_inr_monthly = data_inr_monthly.fillna(method='ffill')

data_gbp_monthly = data_gbp.resample('M').mean().replace(0,np.nan)
data_gbp_monthly = data_gbp_monthly.fillna(method='ffill')

ind_sensex_monthly = ind_sensex.resample('M').mean().replace(0,np.nan)
ind_sensex_monthly = ind_sensex_monthly.fillna(method='ffill')

usd_nifty_monthly = usd_nifty.resample('M').mean().replace(0,np.nan)
usd_nifty_monthly = usd_nifty_monthly.fillna(method='ffill')

gbp_lse_monthly = gbp_lse.resample('M').mean().replace(0,np.nan)
gbp_lse_monthly = gbp_lse_monthly.fillna(method='ffill')

can_tse_monthly = can_tse.resample('M').mean().replace(0,np.nan)
can_tse_monthly = can_tse_monthly.fillna(method='ffill')

ind_inflation = ind_inflation.astype(float).replace(0,np.nan).replace('.',0)
ind_inflation_monthly = ind_inflation.resample('M').mean()
ind_inflation_monthly = ind_inflation_monthly.fillna(method='ffill')

usd_inflation = usd_inflation.replace(0,np.nan).replace('.',0)
usd_inflation = usd_inflation.astype(float)
usd_inflation_monthly = usd_inflation.resample('M').mean()
usd_inflation_monthly = usd_inflation_monthly.fillna(method='ffill')

gbp_inflation = gbp_inflation.astype(float).replace(0,np.nan).replace('.',0)
gbp_inflation_monthly = gbp_inflation.resample('M').mean()
gbp_inflation_monthly = gbp_inflation_monthly.fillna(method='ffill')

can_inflation = can_inflation.astype(float).replace(0,np.nan).replace('.',0)
can_inflation_monthly = can_inflation.resample('M').mean()
can_inflation_monthly = can_inflation_monthly.fillna(method='ffill')

can_gdp = can_gdp.astype(float).replace(0,np.nan).replace('.',0)
can_gdp_monthly = can_gdp.resample('M').mean()
can_gdp_monthly = can_gdp_monthly.fillna(method='ffill')

## Creating a dataframe for monthly data
monthly_data = pd.concat([data_inr_monthly, data_usd_monthly, data_gbp_monthly, ind_sensex_monthly, usd_nifty_monthly, gbp_lse_monthly, can_tse_monthly, ind_inflation_monthly, usd_inflation_monthly, gbp_inflation_monthly, can_inflation_monthly, can_gdp_monthly], axis=1)
monthly_data = monthly_data.astype(float)
monthly_data.drop(['Unnamed: 2'],axis=1,inplace=True)

## Keep monthly_data as backup dataframe
df_monthly = monthly_data.copy()

## Select required columns
df_monthly = df_monthly[['inr_close','usd_close','gbp_close','india_close_sensex', 'nifty_close', 'lse_close', 'tse_close','indinflation', 'usinflation', 'uk_inflation', 'canada_inflation','can_gdp']]

## Feature Engineer - previous month highest closing price
sensexIndia = MinMaxScaler()
df_monthly['scaled_sensex_close'] = sensexIndia.fit_transform(df_monthly[['india_close_sensex']])
niftyUS = MinMaxScaler()
df_monthly['scaled_nifty_close'] = niftyUS.fit_transform(df_monthly[['nifty_close']])
lseUK = MinMaxScaler()
df_monthly['scaled_lse_close'] = lseUK.fit_transform(df_monthly[['lse_close']])
tseCanada = MinMaxScaler()
df_monthly['scaled_india_close'] = tseCanada.fit_transform(df_monthly[['tse_close']])

## Feature Engineer - Manually Normalize inflation, gdp and create a differenced column for inflation rates (can use MinMax/Standard scalar)
df_monthly['norm_indinflation'] = (df_monthly['indinflation']-df_monthly['indinflation'].mean())/df_monthly['indinflation'].std()
df_monthly['norm_usinflation'] = (df_monthly['usinflation']-df_monthly['usinflation'].mean())/df_monthly['usinflation'].std()
df_monthly['norm_uk_inflation'] = (df_monthly['uk_inflation']-df_monthly['uk_inflation'].mean())/df_monthly['uk_inflation'].std()
df_monthly['norm_canada_inflation'] = (df_monthly['canada_inflation']-df_monthly['canada_inflation'].mean())/df_monthly['canada_inflation'].std()
df_monthly['norm_canada_gdp'] = (df_monthly['can_gdp']-df_monthly['can_gdp'].mean())/df_monthly['can_gdp'].std()
df_monthly['diff_norm_ind_cad_inflation'] = df_monthly['norm_canada_inflation'] - df_monthly['norm_indinflation']
df_monthly['diff_norm_usd_cad_inflation'] = df_monthly['norm_canada_inflation'] - df_monthly['norm_usinflation']
df_monthly['diff_norm_gbp_cad_inflation'] = df_monthly['norm_canada_inflation'] - df_monthly['norm_uk_inflation']

## Feature Engineer - Previous month prices and shift to predict next month
df_monthly['inr_close_prev'] = df_monthly['inr_close'].shift(1)
df_monthly['usd_close_prev'] = df_monthly['usd_close'].shift(1)
df_monthly['gbp_close_prev'] = df_monthly['gbp_close'].shift(1)

## Feature Engineer - Rolling average of last quarter to predict next month
df_monthly['indclose_rolling_quarter'] = df_monthly['inr_close'].rolling(window=3, center=False).mean()
df_monthly['usdclose_rolling_quarter'] = df_monthly['usd_close'].rolling(window=3, center=False).mean()
df_monthly['gbpclose_rolling_quarter'] = df_monthly['gbp_close'].rolling(window=3, center=False).mean()

## only for reference, will have to drop too many NaN's if i use this feature
df_monthly['inrclose_rolling_18months'] = df_monthly['inr_close'].rolling(window=18, center=False).mean()
df_monthly['usdclose_rolling_18months'] = df_monthly['usd_close'].rolling(window=18, center=False).mean()
df_monthly['gbpclose_rolling_18months'] = df_monthly['gbp_close'].rolling(window=18, center=False).mean()





###### ------ US Monthly Forecast ------ ####
df_us_monthly = df_monthly[['usd_close','scaled_nifty_close','diff_norm_usd_cad_inflation','usd_close_prev','usdclose_rolling_quarter']]
df_us_monthly.reset_index(inplace=True)
df_us_monthly.columns = ['ds','y','nifty', 'inflation_cad_usd', 'nifty_prev_month', 'price_rollingmean_lastquarter']
df_us_monthly['month'] = df_us_monthly['ds'].dt.month


## Base Graph
fig_base_us_monthly = px.line(df_us_monthly[:-3], x='ds', y='y', hover_data=['nifty_prev_month', 'inflation_cad_usd'], title = ' Monthly Exchange Rates: CAD->USD Ex. 1 CAD = 0.80 USD', labels={"ds": "Month, Year", "y": "Historical Rate", "nifty_prev_month": 'NYSE', 'inflation_cad_usd': 'Monthly Inflation', })
fig_base_us_monthly.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))
fig_base_us_monthly.layout.update(title_x=0.5)
fig_base_us_monthly.update_layout(width=1000, font_color='grey')

fig_usd_monthly_volatality = px.line(data_usd_monthly[:-4], x=data_usd_monthly.index[:-4],y= data_usd_monthly['usd_close'].diff(1)[:-4], hover_data=['usd_open','usd_high','usd_low','usd_close'], title = 'Market Volatality', labels={ 'x': "Year", 'y': "Normalized Price"})
fig_usd_monthly_volatality.update_xaxes(rangeslider_visible=False)
fig_usd_monthly_volatality.update_layout(width=430, height=260,  font_color='grey', margin=dict(r=0.5, l=2, b=0, t=0))
# fig_usd_daily_volatality.layout.update(title = 'sdsadasdsa')
#fig_base_us_daily.show()
## check correlation
#df_us_daily[['y','nifty_prev', 'inflation_cad_usd', 'close_yesterday', 'month']].corr()
def over_80(rate):
    if rate > 0.80:
        return 1
    else:
        return 0
df_us_monthly['high_rate'] = df_us_monthly['y'].apply(over_80)
df_us_monthly['month_bins'] = pd.cut(df_us_monthly['month'], bins=3, labels=False)
## train_test
train_usd_monthly = df_us_monthly[df_us_monthly['ds'] < '2021-05-01'][1:] ##removing NaN's
test_usd_monthly = df_us_monthly[df_us_monthly['ds'] >= '2021-05-01']

future_usd_monthly = model_usd_monthly.make_future_dataframe(periods=5, freq='M')
future_usd_monthly['nifty'] = df_us_monthly['nifty']
future_usd_monthly['inflation_cad_usd'] = df_us_monthly['inflation_cad_usd']
future_usd_monthly['nifty_prev_month'] = df_us_monthly['nifty_prev_month']
future_usd_monthly['price_rollingmean_lastquarter'] = df_us_monthly['price_rollingmean_lastquarter']
future_usd_monthly['high_rate'] = df_us_monthly['high_rate']
future_usd_monthly['month_bins'] = df_us_monthly['month_bins']
forecast_usd_monthly = model_usd_monthly.predict(future_usd_monthly[2:])

multivariate_usd_monthly_test = pd.DataFrame(forecast_usd_monthly.ds[-5:].reset_index(drop=True))
multivariate_usd_monthly_test['Actual'] = test_usd_monthly['y'].reset_index(drop=True)
multivariate_usd_monthly_test['Predicted'] = forecast_usd_monthly[-5:]['yhat'].reset_index(drop=True)

multivariate_usd_monthly_test.index = multivariate_usd_monthly_test.set_index('ds').index.strftime("%Y-%m")
multivariate_usd_monthly_test.drop(['ds'],axis=1, inplace=True)
multivariate_usd_monthly_test = multivariate_usd_monthly_test[1:]

mainplot_usd_monthly = forecast_usd_monthly[['ds', 'trend', 'yhat_lower','yhat', 'yhat_upper']]
mainplot_usd_monthly.set_index('ds', inplace = True)
mainplot_usd_monthly.columns = ['Trend','Lower Limit','Actual','Upper Limit']

fig_usd_monthly_range = px.line(mainplot_usd_monthly, x=mainplot_usd_monthly.index,y= mainplot_usd_monthly['Actual'], hover_data=['Trend','Lower Limit','Actual','Upper Limit'])
fig_usd_monthly_range.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=3, label="20", step="day", stepmode="todate"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))

fig_usd_monthly_range.update_layout(width=630, height=400,  font_color='grey')

us_monthly_analytics = df_us_monthly[['month','y']].groupby('month').agg({'y':{'mean','std'}})



###### ------ INR Monthly Forecast ------ ####

df_inr_monthly = df_monthly[['inr_close','scaled_sensex_close','diff_norm_ind_cad_inflation','inr_close_prev','indclose_rolling_quarter']]
df_inr_monthly.reset_index(inplace=True)
df_inr_monthly.columns = ['ds','y','sensex', 'inflation_cad_inr', 'sensex_prev_month', 'price_rollingmean_lastquarter']
df_inr_monthly['month'] = df_inr_monthly['ds'].dt.month
## Base Graph
fig_base_inr_monthly = px.line(df_inr_monthly[:-3], x='ds', y='y', hover_data=['sensex', 'inflation_cad_inr', 'sensex_prev_month', 'price_rollingmean_lastquarter', 'month'], title = ' Monthly Exchange Rates: CAD->INR Ex. 1 CAD = 58 INR', labels={ "ds": "Time Period (Years 2004 - Present)","y": "Exchange Rate" })

fig_base_inr_monthly.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))
fig_base_inr_monthly.layout.update(title_x=0.5)
fig_base_inr_monthly.update_layout(width=1000, font_color='grey')

fig_inr_monthly_volatality = px.line(data_inr_monthly[:-4], x=data_inr_monthly.index[:-4],y= data_inr_monthly['inr_close'].diff(1)[:-4], hover_data=['inr_open','inr_high','inr_low','inr_close'], title = 'Market Volatality', labels={ 'x': "Year", 'y': "Normalized Price"})
fig_inr_monthly_volatality.update_xaxes(rangeslider_visible=False)
fig_inr_monthly_volatality.update_layout(width=430, height=260,  font_color='grey', margin=dict(r=0.5, l=2, b=0, t=0))
# fig_usd_daily_volatality.layout.update(title = 'sdsadasdsa')
#fig_base_us_daily.show()
## check correlation
#df_us_daily[['y','nifty_prev', 'inflation_cad_usd', 'close_yesterday', 'month']].corr()
def over_80(rate):
    if rate > 0.80:
        return 1
    else:
        return 0
df_inr_monthly['high_rate'] = df_inr_monthly['y'].apply(over_80)
df_inr_monthly['month_bins'] = pd.cut(df_inr_monthly['month'], bins=3, labels=False)
## train_test
train_inr_monthly = df_inr_monthly[df_inr_monthly['ds'] < '2021-05-01'][1:] ##removing NaN's
test_inr_monthly = df_inr_monthly[df_inr_monthly['ds'] >= '2021-05-01']

future_inr_monthly = model_inr_monthly.make_future_dataframe(periods=5, freq='M')
future_inr_monthly['sensex'] = df_inr_monthly['sensex']
future_inr_monthly['inflation_cad_inr'] = df_inr_monthly['inflation_cad_inr']
future_inr_monthly['sensex_prev_month'] = df_inr_monthly['sensex_prev_month']
future_inr_monthly['price_rollingmean_lastquarter'] = df_inr_monthly['price_rollingmean_lastquarter']
future_inr_monthly['high_rate'] = df_inr_monthly['high_rate']
future_inr_monthly['month_bins'] = df_inr_monthly['month_bins']
forecast_inr_monthly = model_inr_monthly.predict(future_inr_monthly[2:])

multivariate_inr_monthly_test = pd.DataFrame(forecast_inr_monthly.ds[-5:].reset_index(drop=True))
multivariate_inr_monthly_test['Actual'] = test_inr_monthly['y'].reset_index(drop=True)
multivariate_inr_monthly_test['Predicted'] = forecast_inr_monthly[-5:]['yhat_lower'].reset_index(drop=True)

multivariate_inr_monthly_test.index = multivariate_inr_monthly_test.set_index('ds').index.strftime("%Y-%m")
multivariate_inr_monthly_test.drop(['ds'],axis=1, inplace=True)
multivariate_inr_monthly_test = multivariate_inr_monthly_test[1:]

mainplot_inr_monthly = forecast_inr_monthly[['ds', 'trend', 'yhat_lower','yhat', 'yhat_upper']]
mainplot_inr_monthly.set_index('ds', inplace = True)
mainplot_inr_monthly.columns = ['Trend','Lower Limit','Actual','Upper Limit']

fig_inr_monthly_range = px.line(mainplot_inr_monthly, x=mainplot_inr_monthly.index,y= mainplot_inr_monthly['Actual'], hover_data=['Trend','Lower Limit','Actual','Upper Limit'])
fig_inr_monthly_range.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=3, label="20", step="day", stepmode="todate"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))

fig_inr_monthly_range.update_layout(width=630, height=400,  font_color='grey')
inr_monthly_analytics = df_inr_monthly[['month','y']].groupby('month').agg({'y':{'mean','std'}})



###### ------ GBP Monthly Forecast ------ ####

df_gbp_monthly = df_monthly[['gbp_close','scaled_lse_close','diff_norm_gbp_cad_inflation','inr_close_prev','gbpclose_rolling_quarter']]
df_gbp_monthly.reset_index(inplace=True)
df_gbp_monthly.columns = ['ds','y','LSE', 'inflation_cad_gbp', 'lse_prev_month', 'price_rollingmean_lastquarter']
df_gbp_monthly['month'] = df_gbp_monthly['ds'].dt.month
## Base Graph

fig_base_gbp_monthly = px.line(df_gbp_monthly[:-3], x='ds', y='y', hover_data=['LSE', 'inflation_cad_gbp', 'lse_prev_month', 'price_rollingmean_lastquarter', 'month'], title = ' Monthly Exchange Rates: CAD->GBP Ex. 1 CAD = 0.59 GBP',labels={ "ds": "Time Period (Years 2004 - Present)","y": "Exchange Rate" })

fig_base_gbp_monthly.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))
fig_base_gbp_monthly.layout.update(title_x=0.5)
fig_base_gbp_monthly.update_layout(width=1000, font_color='grey')

fig_gbp_monthly_volatality = px.line(data_gbp_monthly[:-4], x=data_gbp_monthly.index[:-4],y= data_gbp_monthly['gbp_close'].diff(1)[:-4], hover_data=['gbp_open','gbp_high','gbp_low','gbp_close'], title = 'Market Volatality', labels={ 'x': "Year", 'y': "Normalized Price"})
fig_gbp_monthly_volatality.update_xaxes(rangeslider_visible=False)
fig_gbp_monthly_volatality.update_layout(width=430, height=260,  font_color='grey', margin=dict(r=0.5, l=2, b=0, t=0))
# fig_usd_daily_volatality.layout.update(title = 'sdsadasdsa')
#fig_base_us_daily.show()
## check correlation
#df_us_daily[['y','nifty_prev', 'inflation_cad_usd', 'close_yesterday', 'month']].corr()
def over_80(rate):
    if rate > 0.80:
        return 1
    else:
        return 0
df_gbp_monthly['high_rate'] = df_gbp_monthly['y'].apply(over_80)
df_gbp_monthly['month_bins'] = pd.cut(df_gbp_monthly['month'], bins=3, labels=False)
## train_test
train_gbp_monthly = df_gbp_monthly[df_gbp_monthly['ds'] < '2021-05-01'][1:] ##removing NaN's
test_gbp_monthly = df_gbp_monthly[df_gbp_monthly['ds'] >= '2021-05-01']

future_gbp_monthly = model_gbp_monthly.make_future_dataframe(periods=5, freq='M')
future_gbp_monthly['LSE'] = df_gbp_monthly['LSE']
future_gbp_monthly['inflation_cad_gbp'] = df_gbp_monthly['inflation_cad_gbp']
future_gbp_monthly['lse_prev_month'] = df_gbp_monthly['lse_prev_month']
future_gbp_monthly['price_rollingmean_lastquarter'] = df_gbp_monthly['price_rollingmean_lastquarter']
future_gbp_monthly['high_rate'] = df_gbp_monthly['high_rate']
future_gbp_monthly['month_bins'] = df_gbp_monthly['month_bins']
forecast_gbp_monthly = model_gbp_monthly.predict(future_gbp_monthly[2:])

multivariate_gbp_monthly_test = pd.DataFrame(forecast_gbp_monthly.ds[-5:].reset_index(drop=True))
multivariate_gbp_monthly_test['Actual'] = test_gbp_monthly['y'].reset_index(drop=True)
multivariate_gbp_monthly_test['Predicted'] = forecast_gbp_monthly[-5:]['yhat_lower'].reset_index(drop=True)

multivariate_gbp_monthly_test.index = multivariate_gbp_monthly_test.set_index('ds').index.strftime("%Y-%m")
multivariate_gbp_monthly_test.drop(['ds'],axis=1, inplace=True)
multivariate_gbp_monthly_test = multivariate_gbp_monthly_test[1:]

mainplot_gbp_monthly = forecast_gbp_monthly[['ds', 'trend', 'yhat_lower','yhat', 'yhat_upper']]
mainplot_gbp_monthly.set_index('ds', inplace = True)
mainplot_gbp_monthly.columns = ['Trend','Lower Limit','Actual','Upper Limit']

fig_gbp_monthly_range = px.line(mainplot_gbp_monthly, x=mainplot_gbp_monthly.index,y= mainplot_gbp_monthly['Actual'], hover_data=['Trend','Lower Limit','Actual','Upper Limit'])
fig_gbp_monthly_range.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=3, label="20", step="day", stepmode="todate"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))

fig_gbp_monthly_range.update_layout(width=630, height=400,  font_color='grey')
gbp_monthly_analytics = df_gbp_monthly[['month','y']].groupby('month').agg({'y':{'mean','std'}})

######## --------------------------- DAILY FORECASTS ------------------------- ######
df_daily = daily_data.copy()
df_daily = df_daily[['inr_close','usd_close','gbp_close','india_close_sensex', 'nifty_close', 'lse_close', 'tse_close', 'indinflation', 'usinflation', 'uk_inflation', 'canada_inflation', 'can_gdp']]
## Feature Engineer - previous month highest closing price
sensexIndia = MinMaxScaler()
df_daily['scaled_sensex_close'] = sensexIndia.fit_transform(df_daily[['india_close_sensex']])
niftyUS = MinMaxScaler()
df_daily['scaled_nifty_close'] = niftyUS.fit_transform(df_daily[['nifty_close']])
lseUK = MinMaxScaler()
df_daily['scaled_lse_close'] = lseUK.fit_transform(df_daily[['lse_close']])
tseCanada = MinMaxScaler()
df_daily['scaled_india_close'] = tseCanada.fit_transform(df_daily[['tse_close']])
## Feature Engineer - Manually Normalize inflation, gdp and create a differenced column for inflation rates (can use MinMax/Standard scalar)
df_daily['norm_indinflation'] = (df_daily['indinflation']-df_daily['indinflation'].mean())/df_daily['indinflation'].std()
df_daily['norm_usinflation'] = (df_daily['usinflation']-df_daily['usinflation'].mean())/df_daily['usinflation'].std()
df_daily['norm_uk_inflation'] = (df_daily['uk_inflation']-df_daily['uk_inflation'].mean())/df_daily['uk_inflation'].std()
df_daily['norm_canada_inflation'] = (df_daily['canada_inflation']-df_daily['canada_inflation'].mean())/df_daily['canada_inflation'].std()
df_daily['norm_canada_gdp'] = (df_daily['can_gdp']-df_daily['can_gdp'].mean())/df_daily['can_gdp'].std()
df_daily['diff_norm_ind_cad_inflation'] = df_daily['norm_canada_inflation'] - df_daily['norm_indinflation']
df_daily['diff_norm_usd_cad_inflation'] = df_daily['norm_canada_inflation'] - df_daily['norm_usinflation']
df_daily['diff_norm_gbp_cad_inflation'] = df_daily['norm_canada_inflation'] - df_daily['norm_uk_inflation']
## Feature Engineer - Previous month prices and shift to predict next month
df_daily['inr_close_prev'] = df_daily['inr_close'].shift(1)
df_daily['usd_close_prev'] = df_daily['usd_close'].shift(1)
df_daily['gbp_close_prev'] = df_daily['gbp_close'].shift(1)
df_daily['scaled_nifty_close_prev'] = df_daily['scaled_nifty_close'].shift(1)
df_daily['scaled_sensex_close_prev'] = df_daily['scaled_sensex_close'].shift(1)
df_daily['scaled_lse_close_prev'] = df_daily['scaled_lse_close'].shift(1)
df_daily = df_daily.drop(['indinflation','usinflation','uk_inflation','canada_inflation','india_close_sensex','nifty_close','lse_close','tse_close','can_gdp','norm_indinflation','norm_usinflation','norm_uk_inflation','norm_canada_inflation','scaled_nifty_close', 'scaled_lse_close', 'scaled_india_close'], axis=1)


###### ------ USD Daily Forecast ------ ####
df_us_daily = df_daily[['usd_close','scaled_nifty_close_prev','diff_norm_usd_cad_inflation','usd_close_prev']]
df_us_daily.reset_index(inplace=True)
df_us_daily.columns = ['ds','y','nifty_prev', 'inflation_cad_usd', 'close_yesterday']
df_us_daily['month'] = df_us_daily['ds'].dt.month

## Base Graph
fig_base_us_daily = px.line(round(df_us_daily[:-84],2), x='ds', y='y', hover_data=['nifty_prev', 'inflation_cad_usd', 'close_yesterday', 'month'], title = ' Daily Exchange Rates: CAD->USD Ex. 1 CAD = 0.80 USD', labels={ "ds": "Year  ", "y": "Price  ", 'nifty_prev':'NYSE  ', 'inflation_cad_usd':'Inflation  ', 'close_yesterday':'Previous Close  ' })
fig_base_us_daily.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))
fig_base_us_daily.layout.update(title_x=0.5)
fig_base_us_daily.update_layout(width=1000, font_color='grey')

fig_usd_daily_volatality = px.line(data_usd_daily[:-84], x=data_usd_daily.index[:-84],y= data_usd_daily['usd_close'].diff(1)[:-84], hover_data=['usd_open','usd_high','usd_low','usd_close'], title = 'Market Volatality', labels={ 'x': "Year", 'y': "Normalized Price"})
fig_usd_daily_volatality.update_xaxes(rangeslider_visible=False)
fig_usd_daily_volatality.update_layout(width=430, height=260,  font_color='grey', margin=dict(r=0.5, l=2, b=0, t=0))
# fig_usd_daily_volatality.layout.update(title = 'sdsadasdsa')
#fig_base_us_daily.show()
## check correlation
#df_us_daily[['y','nifty_prev', 'inflation_cad_usd', 'close_yesterday', 'month']].corr()
def over_80(rate):
    if rate > 0.80:
        return 1
    else:
        return 0
df_us_daily['high_rate'] = df_us_daily['y'].apply(over_80)
df_us_daily['month_bins'] = pd.cut(df_us_daily['month'], bins=3, labels=False)
## train_test
train_usd_daily = df_us_daily[df_us_daily['ds'] < '2021-05-01'][1:] ##removing NaN's
test_usd_daily = df_us_daily[df_us_daily['ds'] >= '2021-05-01']

future_usd_daily = model_usd_daily.make_future_dataframe(periods=109, freq='B')
future_usd_daily['nifty_prev'] = df_us_daily['nifty_prev']
future_usd_daily['inflation_cad_usd'] = df_us_daily['inflation_cad_usd']
future_usd_daily['close_yesterday'] = df_us_daily['close_yesterday']
future_usd_daily['high_rate'] = df_us_daily['high_rate']
future_usd_daily['month_bins'] = df_us_daily['month_bins']
forecast_usd_daily = model_usd_daily.predict(future_usd_daily[1:])

multivariate_usd_daily_test = pd.DataFrame(forecast_usd_daily.ds[-109:].reset_index(drop=True))
multivariate_usd_daily_test['Actual'] = test_usd_daily['y'].reset_index(drop=True)
multivariate_usd_daily_test['Predicted'] = forecast_usd_daily[-109:]['yhat'].reset_index(drop=True)

multivariate_usd_daily_test.index = multivariate_usd_daily_test.set_index('ds').index.strftime("%Y-%m-%d")
multivariate_usd_daily_test.drop(['ds'],axis=1, inplace=True)
multivariate_usd_daily_test = multivariate_usd_daily_test[25:]

mainplot_usd_daily = forecast_usd_daily[['ds', 'trend', 'yhat_lower','yhat', 'yhat_upper']]
mainplot_usd_daily.set_index('ds', inplace = True)
mainplot_usd_daily.columns = ['Trend','Lower Limit','Actual','Upper Limit']

fig_usd_daily_range = px.line(mainplot_usd_daily, x=mainplot_usd_daily.index,y= mainplot_usd_daily['Actual'], hover_data=['Trend','Lower Limit','Actual','Upper Limit'])
fig_usd_daily_range.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=3, label="20", step="day", stepmode="todate"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))

fig_usd_daily_range.update_layout(width=630, height=400,  font_color='grey')


###### ------ INR Daily Forecast ------ ####
df_inr_daily = df_daily[['inr_close','scaled_sensex_close_prev','diff_norm_ind_cad_inflation','inr_close_prev']]
df_inr_daily.reset_index(inplace=True)
df_inr_daily.columns = ['ds','y','sensex_prev', 'inflation_cad_inr', 'close_yesterday']
df_inr_daily['month'] = df_inr_daily['ds'].dt.month

## Base Graph
fig_base_inr_daily = px.line(round(df_inr_daily[:-84],2), x='ds', y='y', hover_data=['sensex_prev', 'inflation_cad_inr', 'close_yesterday', 'month'], title = ' Daily Exchange Rates: CAD->INR Ex. 1 CAD = 58 INR', labels={ "ds": "Year  ", "y": "Price  ", 'sensex_prev':'SENSEX  ', 'inflation_cad_inr':'Inflation  ', 'close_yesterday':'Previous Close  ' })
fig_base_inr_daily.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))
fig_base_inr_daily.layout.update(title_x=0.5)
fig_base_inr_daily.update_layout(width=1000, font_color='grey')

fig_inr_daily_volatality = px.line(data_inr_daily[:-84], x=data_inr_daily.index[:-84],y= data_inr_daily['inr_close'].diff(1)[:-84], hover_data=['inr_open','inr_high','inr_low','inr_close'], title = 'Market Volatality', labels={ 'x': "Year", 'y': "Normalized Price"})
fig_inr_daily_volatality.update_xaxes(rangeslider_visible=False)
fig_inr_daily_volatality.update_layout(width=430, height=260,  font_color='grey', margin=dict(r=0.5, l=2, b=0, t=0))
# fig_usd_daily_volatality.layout.update(title = 'sdsadasdsa')
#fig_base_us_daily.show()
## check correlation
#df_us_daily[['y','nifty_prev', 'inflation_cad_usd', 'close_yesterday', 'month']].corr()
def over_80(rate):
    if rate > 0.80:
        return 1
    else:
        return 0
df_inr_daily['high_rate'] = df_inr_daily['y'].apply(over_80)
df_inr_daily['month_bins'] = pd.cut(df_inr_daily['month'], bins=3, labels=False)
## train_test
train_inr_daily = df_inr_daily[df_inr_daily['ds'] < '2021-05-01'][1:] ##removing NaN's
test_inr_daily = df_inr_daily[df_inr_daily['ds'] >= '2021-05-01']

future_inr_daily = model_inr_daily.make_future_dataframe(periods=109, freq='B')
future_inr_daily['sensex_prev'] = df_inr_daily['sensex_prev']
future_inr_daily['inflation_cad_inr'] = df_inr_daily['inflation_cad_inr']
future_inr_daily['close_yesterday'] = df_inr_daily['close_yesterday']
future_inr_daily['high_rate'] = df_inr_daily['high_rate']
future_inr_daily['month_bins'] = df_inr_daily['month_bins']
forecast_inr_daily = model_inr_daily.predict(future_inr_daily[1:])

multivariate_inr_daily_test = pd.DataFrame(forecast_inr_daily.ds[-109:].reset_index(drop=True))
multivariate_inr_daily_test['Actual'] = test_inr_daily['y'].reset_index(drop=True)
multivariate_inr_daily_test['Predicted'] = forecast_inr_daily[-109:]['yhat'].reset_index(drop=True)

multivariate_inr_daily_test.index = multivariate_inr_daily_test.set_index('ds').index.strftime("%Y-%m-%d")
multivariate_inr_daily_test.drop(['ds'],axis=1, inplace=True)
multivariate_inr_daily_test = multivariate_inr_daily_test[25:]

mainplot_inr_daily = forecast_inr_daily[['ds', 'trend', 'yhat_lower','yhat', 'yhat_upper']]
mainplot_inr_daily.set_index('ds', inplace = True)
mainplot_inr_daily.columns = ['Trend','Lower Limit','Actual','Upper Limit']

fig_inr_daily_range = px.line(mainplot_inr_daily, x=mainplot_inr_daily.index,y= mainplot_inr_daily['Actual'], hover_data=['Trend','Lower Limit','Actual','Upper Limit'])
fig_inr_daily_range.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=3, label="20", step="day", stepmode="todate"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))

fig_inr_daily_range.update_layout(width=630, height=400,  font_color='grey')



###### ------ GBP Daily Forecast ------ ####
df_gbp_daily = df_daily[['gbp_close','scaled_lse_close_prev','diff_norm_gbp_cad_inflation','gbp_close_prev']]
df_gbp_daily.reset_index(inplace=True)
df_gbp_daily.columns = ['ds','y','LSE_prev', 'inflation_cad_gbp', 'close_yesterday']
df_gbp_daily['month'] = df_gbp_daily['ds'].dt.month
## Base Graph
fig_base_gbp_daily = px.line(round(df_gbp_daily[:-84],2), x='ds', y='y', hover_data=['LSE_prev', 'inflation_cad_gbp', 'close_yesterday', 'month'], title = ' Daily Exchange Rates: CAD->GBP Ex. 1 CAD = 0.59 GBP', labels={ "ds": "Year  ", "y": "Price  ", 'LSE_prev':'FTSE  ', 'inflation_cad_gbp':'Inflation  ', 'close_yesterday':'Previous Close  ' })
fig_base_gbp_daily.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))
fig_base_gbp_daily.layout.update(title_x=0.5)
fig_base_gbp_daily.update_layout(width=1000, font_color='grey')

fig_gbp_daily_volatality = px.line(data_gbp_daily[:-84], x=data_gbp_daily.index[:-84],y= data_gbp_daily['gbp_close'].diff(1)[:-84], hover_data=['gbp_open','gbp_high','gbp_low','gbp_close'], title = 'Market Volatality', labels={ 'x': "Year", 'y': "Normalized Price"})
fig_gbp_daily_volatality.update_xaxes(rangeslider_visible=False)
fig_gbp_daily_volatality.update_layout(width=430, height=260,  font_color='grey', margin=dict(r=0.5, l=2, b=0, t=0))
# fig_usd_daily_volatality.layout.update(title = 'sdsadasdsa')
#fig_base_us_daily.show()
## check correlation
#df_us_daily[['y','nifty_prev', 'inflation_cad_usd', 'close_yesterday', 'month']].corr()
def over_80(rate):
    if rate > 0.80:
        return 1
    else:
        return 0
df_gbp_daily['high_rate'] = df_gbp_daily['y'].apply(over_80)
df_gbp_daily['month_bins'] = pd.cut(df_gbp_daily['month'], bins=3, labels=False)
## train_test
train_gbp_daily = df_gbp_daily[df_gbp_daily['ds'] < '2021-05-01'][1:] ##removing NaN's
test_gbp_daily = df_gbp_daily[df_gbp_daily['ds'] >= '2021-05-01']

future_gbp_daily = model_gbp_daily.make_future_dataframe(periods=109, freq='B')
future_gbp_daily['LSE_prev'] = df_gbp_daily['LSE_prev']
future_gbp_daily['inflation_cad_gbp'] = df_gbp_daily['inflation_cad_gbp']
future_gbp_daily['close_yesterday'] = df_gbp_daily['close_yesterday']
future_gbp_daily['high_rate'] = df_gbp_daily['high_rate']
future_gbp_daily['month_bins'] = df_gbp_daily['month_bins']
forecast_gbp_daily = model_gbp_daily.predict(future_gbp_daily[1:])

multivariate_gbp_daily_test = pd.DataFrame(forecast_gbp_daily.ds[-109:].reset_index(drop=True))
multivariate_gbp_daily_test['Actual'] = test_gbp_daily['y'].reset_index(drop=True)
multivariate_gbp_daily_test['Predicted'] = forecast_gbp_daily[-109:]['yhat'].reset_index(drop=True)

multivariate_gbp_daily_test.index = multivariate_gbp_daily_test.set_index('ds').index.strftime("%Y-%m-%d")
multivariate_gbp_daily_test.drop(['ds'],axis=1, inplace=True)
multivariate_gbp_daily_test = multivariate_gbp_daily_test[25:]

mainplot_gbp_daily = forecast_usd_daily[['ds', 'trend', 'yhat_lower','yhat', 'yhat_upper']]
mainplot_gbp_daily.set_index('ds', inplace = True)
mainplot_gbp_daily.columns = ['Trend','Lower Limit','Actual','Upper Limit']

fig_gbp_daily_range = px.line(mainplot_gbp_daily, x=mainplot_gbp_daily.index,y= mainplot_gbp_daily['Actual'], hover_data=['Trend','Lower Limit','Actual','Upper Limit'])
fig_gbp_daily_range.update_xaxes(rangeslider_visible=False, rangeselector=dict( buttons=list([
            dict(count=3, label="20", step="day", stepmode="todate"),
            dict(count=2, label="2y", step="year", stepmode="todate"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")])))

fig_gbp_daily_range.update_layout(width=630, height=400,  font_color='grey')
#---------------------------------#
# Title

image = Image.open('logo3.png')
usa_daily_components_breakdown_image = Image.open('./Images/02_USD_Daily_Component_Plots.png')
inr_daily_components_breakdown_image = Image.open('./Images/03_INR_Daily_Component_Plots.png')
gbp_daily_components_breakdown_image = Image.open('./Images/04_GBP_Daily_Component_Plots.png')
forex_ref_image = Image.open('./Images/rupee-dollar.jpeg')
# st.image(image, width = 300)
#
# st.title('Forex Predictor App')
# st.markdown("""
# This app interconverts the value of foreign currencies!
#
# """)

#---------------------------------#
# Sidebar + Main panel

st.markdown("""
<style>
sidebar .sidebar-content {
    background-color: #4267B2 !important;
}
</style>
    """, unsafe_allow_html=True)
#
st.sidebar.image(image, width = 300)
add_space = """     <br>    """
st.sidebar.markdown(add_space, unsafe_allow_html=True)

st.sidebar.header('Predict Exchange Rate against CAD,')
#
# ## Sidebar - Currency price unit
# selected_currency = st.sidebar.selectbox('Select Currency', ['CAD to USD Daily','CAD to INR Daily','CAD to GBP Daily','CAD to USD Monthly', 'CAD to INR Monthly',  'CAD to GBP Monthly'])
# # prediction_type = st.sidebar.selectbox('Type of Prediction', ['none','Daily','Monthly'])
selected_currency = st.sidebar.selectbox('Select Currency', ['Step 1: Select Currency','CAD to USD Daily','CAD to INR Daily','CAD to GBP Daily','CAD to USD Monthly', 'CAD to INR Monthly',  'CAD to GBP Monthly'])

@st.cache(suppress_st_warning=True)
def load_data(selected_currency):
    ## Sidebar - Currency price unit
    # prediction_type = st.sidebar.selectbox('Type of Prediction', ['none','Daily','Monthly'])
    if selected_currency == 'CAD to USD Daily':
        period = st.sidebar.slider('Slide to select # of Days:', 1, 20)
        data_usd_daily.index = data_usd_daily.index.strftime("%Y-%m-%d")
        st.title('CAD to USD Current Market,')
        st.plotly_chart(fig_base_us_daily)
        cols = st.beta_columns(2)
        cols[0].write(data_usd_daily.sort_index(ascending=False)[84:])
        cols[1].plotly_chart(fig_usd_daily_volatality)
        cols[1].markdown(""" <grey><center> Volatality of USD Market (2004-2021) </center></grey> """, unsafe_allow_html= True)
        st.markdown(add_space, unsafe_allow_html=True)
        st.title('CAD to USD Forecasts,')
        cols = st.beta_columns(2)
        cols[0].write(multivariate_usd_daily_test['Predicted'][:period])
        cols[1].markdown(""" This prediction was obtained using fbprophet <br> machine learning algorithm. <br> NYSE, Yesterday Price, Inflation and  <br> GDP against Canada were considered for obtaining prediction.""", unsafe_allow_html=True)
        st.title('Explore Forecast with Plots to gain more insights,')
        st.image(usa_daily_components_breakdown_image, width=700)

    elif selected_currency == 'CAD to INR Daily':
        period = st.sidebar.slider('Slide to select # of Days:', 1, 20)
        data_inr_daily.index = data_inr_daily.index.strftime("%Y-%m-%d")
        st.title('CAD to INR Current Market,')
        st.plotly_chart(fig_base_inr_daily)
        cols = st.beta_columns(2)
        cols[0].write(data_inr_daily.sort_index(ascending=False)[84:])
        cols[1].plotly_chart(fig_inr_daily_volatality)
        cols[1].markdown(""" <grey><center> Volatality of INR Market (2004-2021) </center></grey> """, unsafe_allow_html= True)
        st.markdown(add_space, unsafe_allow_html=True)
        st.title('CAD to INR Forecasts,')
        cols = st.beta_columns(2)
        cols[0].write(multivariate_inr_daily_test['Predicted'][:period])
        cols[1].markdown(""" This prediction was obtained using fbprophet <br> machine learning algorithm. <br> SENSEX, Yesterday Price, Inflation and  <br> GDP against Canada were considered for obtaining prediction.""", unsafe_allow_html=True)
        st.title('Explore Forecast with Plots to gain more insights,')
        st.image(inr_daily_components_breakdown_image, width=700)

    elif selected_currency == 'CAD to GBP Daily':
        period = st.sidebar.slider('Slide to select # of Days:', 1, 20)
        data_gbp_daily.index = data_gbp_daily.index.strftime("%Y-%m-%d")
        st.title('CAD to GBP Current Market,')
        st.plotly_chart(fig_base_gbp_daily)
        cols = st.beta_columns(2)
        cols[0].write(data_gbp_daily.sort_index(ascending=False)[84:])
        cols[1].plotly_chart(fig_gbp_daily_volatality)
        cols[1].markdown(""" <grey><center> Volatality of GBP Market (2004-2021) </center></grey> """, unsafe_allow_html= True)
        st.markdown(add_space, unsafe_allow_html=True)
        st.title('CAD to GBP Forecasts,')
        cols = st.beta_columns(2)
        cols[0].write(multivariate_gbp_daily_test['Predicted'][:period])
        cols[1].markdown(""" This prediction was obtained using fbprophet <br> machine learning algorithm. <br> LSE, Yesterday Price, Inflation and  <br> GDP against Canada were considered for obtaining prediction.""", unsafe_allow_html=True)
        st.title('Explore Forecast with Plots to gain more insights,')
        st.image(gbp_daily_components_breakdown_image, width=700)



    elif selected_currency == 'CAD to USD Monthly':
        period = st.sidebar.slider('Slide to select # of Months:', 1, 4)
        data_usd_monthly.index = data_usd_monthly.index.strftime("%Y-%m")
        st.title('CAD to USD Current Market,')
        st.plotly_chart(fig_base_us_monthly)
        cols = st.beta_columns(2)
        cols[0].write(data_usd_monthly.sort_index(ascending=False)[4:])
        cols[1].plotly_chart(fig_usd_monthly_volatality)
        cols[1].markdown(""" <grey><center> Volatality of USD Market (2004-2021) </center></grey> """, unsafe_allow_html= True)
        st.markdown(add_space, unsafe_allow_html=True)
        st.title('CAD to USD Forecasts,')
        cols = st.beta_columns(2)
        cols[0].write(multivariate_usd_monthly_test['Predicted'][:period])
        cols[1].markdown(""" This prediction was obtained using fbprophet <br> machine learning algorithm. <br> NYSE, Yesterday Price, Inflation and  <br> GDP against Canada were considered for obtaining prediction.""", unsafe_allow_html=True)
        st.title('Explore Forecast with Plots to gain more insights,')
        st.image(usa_daily_components_breakdown_image, width=700)


    elif selected_currency == 'CAD to INR Monthly':
        period = st.sidebar.slider('Slide to select # of Months:', 1, 4)
        data_inr_monthly.index = data_inr_monthly.index.strftime("%Y-%m-%d")
        st.title('CAD to INR Current Market,')
        st.plotly_chart(fig_base_inr_monthly)
        cols = st.beta_columns(2)
        cols[0].write(data_inr_monthly.sort_index(ascending=False)[4:])
        cols[1].plotly_chart(fig_inr_monthly_volatality)
        cols[1].markdown(""" <grey><center> Volatality of INR Market (2004-2021) </center></grey> """, unsafe_allow_html= True)
        st.markdown(add_space, unsafe_allow_html=True)
        st.title('CAD to INR Forecasts,')
        cols = st.beta_columns(2)
        cols[0].write(multivariate_inr_monthly_test['Predicted'][:period])
        cols[1].markdown(""" This prediction was obtained using fbprophet <br> machine learning algorithm. <br> SENSEX, Yesterday Price, Inflation and  <br> GDP against Canada were considered for obtaining prediction.""", unsafe_allow_html=True)
        st.title('Explore Forecast with Plots to gain more insights,')
        st.image(inr_monthly_components_breakdown_image, width=700)

    elif selected_currency == 'CAD to GBP Monthly':
        period = st.sidebar.slider('Slide to select # of Months:', 1, 4)
        data_gbp_monthly.index = data_gbp_monthly.index.strftime("%Y-%m-%d")
        st.title('CAD to GBP Current Market,')
        st.plotly_chart(fig_base_gbp_monthly)
        cols = st.beta_columns(2)
        cols[0].write(data_gbp_monthly.sort_index(ascending=False)[4:])
        cols[1].plotly_chart(fig_gbp_monthly_volatality)
        cols[1].markdown(""" <grey><center> Volatality of GBP Market (2004-2021) </center></grey> """, unsafe_allow_html= True)
        st.markdown(add_space, unsafe_allow_html=True)
        st.title('CAD to GBP Forecasts,')
        cols = st.beta_columns(2)
        cols[0].write(multivariate_gbp_monthly_test['Predicted'][:period])
        cols[1].markdown(""" This prediction was obtained using fbprophet <br> machine learning algorithm. <br> LSE, Yesterday Price, Inflation and  <br> GDP against Canada were considered for obtaining prediction.""", unsafe_allow_html=True)
        st.title('Explore Forecast with Plots to gain more insights,')
        st.image(gbp_daily_components_breakdown_image, width=700)

    else:
        st.markdown("""
            <h1 style="color:#4267B2;"> Forex Predictor Application </h1><br>
            <h2>How it works,</h2>
            <h4>Step1: From the left panel, please select approprite Currency and Frequency for Prediction.</h4>
            <h4>Step2: Select Currency, Start and End dates to obtain Prediction</h4> <br>
        """, unsafe_allow_html= True)
        st.markdown("""
            <br><h6><i>uncollapse to know more about the app!</i></h6>
        """, unsafe_allow_html= True)






# # Sidebar
# st.sidebar.subheader('Date to obtain best prediction for,')
# start_date = st.sidebar.date_input("Start date", datetime.date(2021, 6, 7))
# start_date = time.strptime(start_date,"%d/%m/%Y")
# end_date = st.sidebar.date_input("End date", datetime.date(2021, 9, 30))
# end_date = time.strptime(end_date,"%d/%m/%Y")



def check_submit(start_date, end_date, selected_currency):
    if (selected_currency == 'CAD to USD Daily') & ((pd.to_datetime('2021-6-4') < pd.to_datetime(start_date) < pd.to_datetime('2021-9-30'))) & ((pd.to_datetime('2021-6-4') < pd.to_datetime(end_date) < pd.to_datetime('2021-9-30'))) & (pd.to_datetime(start_date) < pd.to_datetime(end_date)):
        forecast_usd_daily2 = forecast_usd_daily[-84:][['ds','yhat']]
        temp_df = forecast_usd_daily2[(forecast_usd_daily2['ds'] > start_date) & (forecast_usd_daily2['ds'] <= end_date)]
        temp_value = np.min(temp_df['yhat'].values)
        day = temp_df[temp_df['yhat']==temp_value]['ds'].iloc[0]
        value = round(temp_df[temp_df['yhat']==temp_value]['yhat'].iloc[0],4)

    elif (selected_currency == 'CAD to INR Daily') & ((pd.to_datetime('2021-6-4') < pd.to_datetime(start_date) < pd.to_datetime('2021-9-30'))) & ((pd.to_datetime('2021-6-4') < pd.to_datetime(end_date) < pd.to_datetime('2021-9-30'))) & (pd.to_datetime(start_date) < pd.to_datetime(end_date)):
        forecast_inr_daily2 = forecast_inr_daily[-84:][['ds','yhat']]
        temp_df = forecast_inr_daily2[(forecast_inr_daily2['ds'] > start_date) & (forecast_inr_daily2['ds'] <= end_date)]
        temp_value = np.min(temp_df['yhat'].values)
        day = temp_df[temp_df['yhat']==temp_value]['ds'].iloc[0]
        value = round(temp_df[temp_df['yhat']==temp_value]['yhat'].iloc[0],4)

    elif (selected_currency == 'CAD to GBP Daily') & ((pd.to_datetime('2021-6-4') < pd.to_datetime(start_date) < pd.to_datetime('2021-9-30'))) & ((pd.to_datetime('2021-6-4') < pd.to_datetime(end_date) < pd.to_datetime('2021-9-30'))) & (pd.to_datetime(start_date) < pd.to_datetime(end_date)):
        forecast_gbp_daily2 = forecast_gbp_daily[-84:][['ds','yhat']]
        temp_df = forecast_gbp_daily2[(forecast_gbp_daily2['ds'] > start_date) & (forecast_gbp_daily2['ds'] <= end_date)]
        temp_value = np.min(temp_df['yhat'].values)
        day = temp_df[temp_df['yhat']==temp_value]['ds'].iloc[0]
        value = round(temp_df[temp_df['yhat']==temp_value]['yhat'].iloc[0],4)

    elif (selected_currency == 'CAD to USD Monthly') & ((pd.to_datetime('2021-6-4') < pd.to_datetime(start_date) < pd.to_datetime('2021-9-30'))) & ((pd.to_datetime('2021-6-4') < pd.to_datetime(end_date) < pd.to_datetime('2021-9-30'))) & (pd.to_datetime(start_date) < pd.to_datetime(end_date)):
        forecast_usd_monthly2 = forecast_usd_monthly[-4:][['ds','yhat']]
        temp_df = forecast_usd_monthly2[(forecast_usd_monthly2['ds'] > start_date) & (forecast_usd_monthly2['ds'] <= end_date)]
        temp_value = np.min(temp_df['yhat'].values)
        day = temp_df[temp_df['yhat']==temp_value]['ds'].iloc[0]
        value = round(temp_df[temp_df['yhat']==temp_value]['yhat'].iloc[0],4)

    elif (selected_currency == 'CAD to INR Monthly') & ((pd.to_datetime('2021-6-4') < pd.to_datetime(start_date) < pd.to_datetime('2021-9-30'))) & ((pd.to_datetime('2021-6-4') < pd.to_datetime(end_date) < pd.to_datetime('2021-9-30'))) & (pd.to_datetime(start_date) < pd.to_datetime(end_date)):
        forecast_inr_monthly2 = forecast_inr_monthly[-4:][['ds','yhat']]
        temp_df = forecast_inr_monthly2[(forecast_inr_monthly2['ds'] > start_date) & (forecast_inr_monthly2['ds'] <= end_date)]
        temp_value = np.min(temp_df['yhat'].values)
        day = temp_df[temp_df['yhat']==temp_value]['ds'].iloc[0]
        value = round(temp_df[temp_df['yhat']==temp_value]['yhat'].iloc[0],4)

    elif (selected_currency == 'CAD to GBP Monthly') & ((pd.to_datetime('2021-6-4') < pd.to_datetime(start_date) < pd.to_datetime('2021-9-30'))) & ((pd.to_datetime('2021-6-4') < pd.to_datetime(end_date) < pd.to_datetime('2021-9-30'))) & (pd.to_datetime(start_date) < pd.to_datetime(end_date)):
        forecast_gbp_monthly2 = forecast_gbp_monthly[-4:][['ds','yhat']]
        temp_df = forecast_gbp_monthly2[(forecast_gbp_monthly2['ds'] > start_date) & (forecast_gbp_monthly2['ds'] <= end_date)]
        temp_value = np.min(temp_df['yhat'].values)
        day = temp_df[temp_df['yhat']==temp_value]['ds'].iloc[0]
        value = round(temp_df[temp_df['yhat']==temp_value]['yhat'].iloc[0],4)

    else:
        day = 'Prediction not available'
        value = 'Please select proper date'

    return day, value


def main():
    st.write('Please select Currency and type of prediction!')
load_data(selected_currency)

    #df = load_data()

st.sidebar.subheader('Date to obtain best prediction for,')
start_date = st.sidebar.date_input("Start date", datetime.date(2021, 6, 7))
start_date = str(start_date)
end_date = st.sidebar.date_input("End date", datetime.date(2021, 9, 30))
end_date = str(end_date)
ds = datetime.datetime.strptime(start_date,"%Y-%m-%d")
ds = ds.strftime("%Y-%m-%d")
de = datetime.datetime.strptime(end_date,"%Y-%m-%d")
de = de.strftime("%Y-%m-%d")
day, value = check_submit(ds, de, selected_currency)
st.sidebar.write('Best Date: ', day)
st.sidebar.write('Best Price: ', value)

#st.markdown(""" """)





#st.write(df)

#st.write( df.transpose() )

#---------------------------------#
# About
expander_bar = st.beta_expander("About")
expander_bar.markdown("""
* **Forex Predictor:** This application mainly targets studends and immigrants who are always looking for a better exchange rates, to save some money just by knowing if the exchange rate is going to be much lower in coming days or months.

* **Libraries used:** numpy, pandas, matplotlib, sklearn, fbprophet, plotly, PIL, pickle, streamlit.
* **Data Sources:** https://ca.finance.yahoo.com/, https://fred.stlouisfed.org/ and https://www150.statcan.gc.ca
* **Reference for Streamlit Web App:** [Nachiketa Hebbar] - Deploy ML model on Webpage| Python (Streamlit) #1| (https://www.youtube.com/channel/UCJPihOKkiT7TqP7NK9-GtuQ_) & [Chanin Nantasenamat](http://twitter.com/thedataprof) (aka [Data Professor](http://youtube.com/dataprofessor))

""")



##* **Python libraries:** streamlit, pandas, pillow, requests, json
##* **Credit:** App built by [Chanin Nantasenamat](http://twitter.com/thedataprof) (aka [Data Professor](http://youtube.com/dataprofessor))
