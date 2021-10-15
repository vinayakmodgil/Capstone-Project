#!/usr/bin/env python
# coding: utf-8

# ## Capstone Project
# 
# >- Name: Vinayak Modgil<br>
# >- Student pace: Full time<br>
# >- Scheduled Project review date: <br>
# >- Instructor Name: Yish Lim<br>
# >- Github URL: 

# ## Predicting Indian Stock Market Prices
# <br>
# by Vinayak Modgil
# <br>
# 

# ## Project Description and Business Statement
# 
# What I am trying to accomplish with the project
# 1. Build an ML model and NN model that will predict whether a stock price will increase or decrease the next day
# 2. Prediction based on the closing price exchanged by the end of a certain day.
# 3. The price of a stock at the end of the day is different than at the beginning of the next day. I want to retreive this difference and figure out if stock price will go up or down based on previous data.
# 4. If model predicts that stock price will go up, buy at the end of the day and sell right away the next day.
# 5. Model will not be even close to 100% accuracy as this is an impossible feat. However, an accuracy of over 60% will give a net gain in the long run.

# ## Analysis and Methodology

# > - [Data Collection](#Data-Collection)
# > - [Feature Evaluation](#Feature-Evaluation)
# > - [Reliance](#RELIANCE)
# > - [HDFCBANK](#HDFC-BANK)
# > - [Global Function](#Global-function)
# > - [ADANIPLORTS](#ADANIPORTS)
# > - [POWERGRID](#POWERGRID)
# > - [ICICIBANK](#ICICI-BANK)
# > - [KOTAKBANK](#KOTAK-BANK)
# > - [ASTRAL](#ASTRAL)
# > - [SUMCHEM](#SUMCHEM)
# > - [BERGEPAINT](#BERGEPAINT)
# > - [IGL](#IGL)
# > - [KANSAINER](#KANSAINER)
# > - [RELAXO](#RELAXO)
# > - [CDSL](#CDSL)
# > - [HDFCLIFE](#HDFCLIFE)
# > - [IEX](#IEX)
# > - [SIEMENS](#SIEMENS)
# > - [CAMS](#CAMS)
# > - [PRINCEPIPE](#PRINCEPIPE)
# > - [SBICARD](#SBICARD)
# > - [ROUTE](#ROUTE)
# 

# ## Data Collection

# In[43]:


#General packages
import os
import random
import warnings 
warnings.filterwarnings("ignore")
from random import gauss

#Data wrangling packages
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns

#Statistical analysis package
import statsmodels.tsa.api as tsa
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Machine Learning packages
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

#More sklearn packages
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

#Metrics packages
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#Deep Learning packages
from keras import models
from keras.models import Sequential
from keras.layers import LSTM, Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

#Hyperparameter tuning
import talos
from sklearn.model_selection import GridSearchCV

#Dl model understanding
import lime
from lime import lime_image
from lime import lime_base
from skimage.segmentation import mark_boundaries

# Created functions
import time_series_functions as tsf
import fbprophet as Prophet
import stock_prediction_algorithm as spa


# In[44]:


#Import the first data
capstone_data = pd.read_csv("stock market/Nifty50_target_data.csv")
capstone_data


# In[45]:


def get_data(csv_list, folder1=None):
    '''
    Merges a list of csv files and
    returns a new DataFrame containing
    all the csv files in the folder
    
    get_data -> (Args)
    csv_list: A list of csv files
    folder: Folder containing all the files
    '''
    if folder1 != None:
        #Specify the path to csv files
        stock_list = os.listdir(folder1)
        #Initialize a DataFrame
        csv_merged = pd.DataFrame()
        for stock in stock_list:
           #List of files to be merged
           csv_list.append(pd.read_csv(folder1 +"/" + stock))
        
           for csv_file in csv_list:
            
            #Add the elements of csv_list to csv_merged
               csv_merged = csv_merged.append(csv_file, ignore_index=True)
            
        # Export the data to the folder
               csv_merged.to_csv("NSE.csv", index=False)
        return csv_merged


# In[46]:


NSE_data = get_data([], folder1="stock market/NSE")


# In[47]:


NSE_data


# In[48]:


nifty_data = get_data([], folder1="stock market/Nifty50")


# In[49]:


nifty_data["Date"] = pd.to_datetime(nifty_data["Date"])
nifty_data.set_index(nifty_data["Date"], inplace=True)
nifty_data.drop("Date",axis=1, inplace=True)
nifty_data


# In[50]:


nifty_data["Close"] = nifty_data["Close"].astype("float")
nifty_data["Open"] = nifty_data["Close"].astype("float")
nifty_data["High"] = nifty_data["Close"].astype("float")
nifty_data["Low"] = nifty_data["Close"].astype("float")
nifty_data["Adj Close"] = nifty_data["Close"].astype("float")
nifty_data["Volume"] = nifty_data["Close"].astype("float")


# In[51]:


NSE_data["Close"].fillna(method="bfill", inplace=True)
NSE_data["Open"].fillna(method="bfill", inplace=True)
NSE_data["High"].fillna(method="bfill", inplace=True)
NSE_data["Low"].fillna(method="bfill", inplace=True)
NSE_data["Adj Close"].fillna(method="bfill", inplace=True)
NSE_data["Volume"].fillna(method="bfill", inplace=True)


# In[178]:


faang = get_data([], folder1="stock market/FAANG")


# In[181]:


faang["Date"] = pd.to_datetime(faang["Date"])
faang.set_index(faang["Date"], inplace=True)
faang.drop("Date", axis=1, inplace=True)


# In[177]:


faang.info()


# In[52]:


NSE_data.info()


# In[53]:


#Display the info of capstone data
capstone_data.info()


# In[54]:


# Descriptive stats of capstone data
capstone_data.describe()


# In[55]:


NSE_data.describe()


# In[196]:


faang.fillna(method="bfill", inplace=True)


# ### Data Schema (capstone_data)
# > - **Prev Close:** closing price of the previous trading day.
# > - **Open:** first price at which a stock trades during regular market hours.
# > - **High:** highest price stock reaches during market hours
# > - **Low:** lowest price stock reaches during market hours
# > - **Last:** final quoted trading price.
# > - **Close:** closing price of the current trading day
# > - **VWAP:** or volume-weighted average price. the average price a stock has traded at throughout the day, based on both volume and price.
# > - **Volume:** the total number of shares that are actually traded.
# > - **Turnover:** calculated by dividing the total number of shares traded during some period by the average number of shares outstanding for the same period.
# > - **Trades:**  buying and selling shares in the secondary market on the same day.
# > - **Deliverable Volume:** quantity of shares which actually move from one set of people (who had those shares in their demat account before today and are selling today)  to another set of people
# > - **%Deliverable:**  Deliverable percentage is (Deliverable quantity / Traded quantity)

# ### Data Schema (NSE_data)
# > - **Open:** first price at which a stock trades during regular market 
# > - **High:** highest price stock reaches during market hours
# > - **Low:** lowest price stock reaches during market hours
# > - **Close:** closing price of the current trading day
# > - **Adj Close:** adjusted closing price amends a stock's closing price to reflect that stock's value after accounting.
# > - **Volume:** the total number of shares that are actually traded.

# ### Data Schema (nifty_data)
# > - **Open:** first price at which a stock trades during regular market 
# > - **High:** highest price stock reaches during market hours
# > - **Low:** lowest price stock reaches during market hours
# > - **Close:** closing price of the current trading day
# > - **Adj Close:** adjusted closing price amends a stock's closing price to reflect that stock's value after accounting.
# > - **Volume:** the total number of shares that are actually traded.

# ### Data Schema (faang)
# > - **Open:** first price at which a stock trades during regular market 
# > - **High:** highest price stock reaches during market hours
# > - **Low:** lowest price stock reaches during market hours
# > - **Close:** closing price of the current trading day
# > - **Adj Close:** adjusted closing price amends a stock's closing price to reflect that stock's value after accounting.
# > - **Volume:** the total number of shares that are actually traded.

# In[56]:


capstone_data["Symbol"].value_counts()


# In[57]:


def extract_data(df, no_of_symbols=0):
    '''
    Extracts the exact amount of stocks
    symbols required for EDA later.
    
    extract_data -> (Args)
    df - A DataFrame
    no_of_symbols - An int
    '''
    if no_of_symbols != 0:
        
        #Initialize an empty symbol_list
        symbol_list = []
        for symbol in df["Symbol"]:
            
            #Use break when the length of
            # symbol_list is equal to 
            # no_of_symbols
            if len(symbol_list) == no_of_symbols:
                break
                
            if  symbol not in symbol_list:
                #Append the symbol to the
                #symbol_list
                symbol_list.append(symbol)
    return symbol_list


# In[58]:


#Extract rows needed for exploration
symbol_list = extract_data(capstone_data, no_of_symbols=50)
symbol_list = [i for i in symbol_list if i == "RELIANCE" or i =="POWERGRID" or i=="HDFCBANK" or i =="ICICIBANK" or i == "KOTAKBANK" or i == "ADANIPORTS"]


# In[59]:


#Change the index to "Symbol" to extract relevant dataset
capstone_data = capstone_data.set_index("Symbol").loc[symbol_list]


# In[60]:


#Set the date to datetime
capstone_data["Date"] = pd.to_datetime(capstone_data["Date"])

#Reset the index of the dataset
capstone_data.reset_index(inplace=True)

#Set the index to date for time series analysis
capstone_data.set_index("Date", inplace=True)


# In[61]:


capstone_data["Symbol"].value_counts()


# In[62]:


NSE_data["Date"] = pd.to_datetime(NSE_data["Date"])
NSE_data.set_index("Date", inplace=True)


# In[63]:


NSE_data


# In[64]:


NSE_data["Symbol"].value_counts()


# In[65]:


capstone_data.isnull().sum()


# In[66]:


cols_to_impute = ["Trades", "Deliverable Volume", "%Deliverble"]
impute_const_int = SimpleImputer(strategy="constant", missing_values=np.nan, fill_value=0)
capstone_data[cols_to_impute] = impute_const_int.fit_transform(capstone_data[cols_to_impute])


# In[67]:


capstone_data.isnull().sum()


# ## Stock Evaluation

# In[68]:


#White noise has mean=0, std=1
mean= 0
std = 1

#Make a white noise time series
noise = pd.Series([gauss(mean, std) for x in range(1000)])
noise.plot(title="White Noise", figsize=(15, 8))

## Check out mean and variance
noise.mean(), noise.var()


# In[69]:


NSE_data["Symbol"].value_counts()


# In[70]:


nifty_data["Symbol"].value_counts()


# ### RELIANCE

# In[71]:


rel = nifty_data.loc[nifty_data["Symbol"] == "RELIANCE"]


# In[72]:


rel["Close"].fillna(method="bfill", inplace=True)
rel["Open"].fillna(method="bfill", inplace=True)
rel["High"].fillna(method="bfill", inplace=True)
rel["Low"].fillna(method="bfill", inplace=True)
rel["Adj Close"].fillna(method="bfill", inplace=True)
rel["Volume"].fillna(method="bfill", inplace=True)


# In[73]:


#Plot the heatmap of the dataset
plt.figure(figsize=(20, 15))
sns.heatmap(rel.corr(), annot=True, cmap="Blues")


# #### Target Variable Evaluation

# In[74]:


rel_close = rel[["Close"]]
rel_close.plot(figsize=(15, 8))


# In[75]:


rel_monthly_mean = rel_close.resample("M").mean()


# In[76]:


rel_monthly_mean.plot(figsize=(15, 6))


# In[77]:


tsf.stationarity_check(rel_monthly_mean)


# In[78]:


## Before 26-11-2009, split the stock price by 1/4
## Before 07-09-2017, split the stock price by 1/2
# rel_close[:"2009-11-27"] = rel_close[:"2009-11-27"] / 4
# rel_close["2009-11-27":"2017-09-08"] = rel_close["2009-11-27":"2017-09-08"] / 2


# In[79]:


rel_close_decomp = seasonal_decompose(rel_monthly_mean, model="multiplicative", freq=30)
fig = plt.figure()
fig = rel_close_decomp.plot()
fig.set_size_inches(16, 9)


# In[80]:


tsf.plot_autocorrelation(rel_monthly_mean, verbose=1)


# In[81]:


rcParams["figure.figsize"] = 15, 6
rel_close_log = np.log(rel_monthly_mean)
moving_avg_rel = rel_close_log.rolling(12).mean()
std_rel = rel_close_log.rolling(12).std()
plt.plot(moving_avg_rel, color="red", label="Rel Mean")
plt.plot(std_rel, color="black", label="Rel Standard Deviation")
plt.title("Moving Average")
plt.legend(loc="best")
plt.show()


# #### Reliance Model

# In[82]:


rel_train = rel_close_log[:int(len(rel_close_log) *0.8)]
rel_test = rel_close_log[int(len(rel_close_log) *0.8):]
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(np.exp(rel_test), color="green", label="Actual Stock Price test")
ax.plot(np.exp(rel_train), label="Actual Stock Price train")
ax.set_xlabel("Dates")
ax.set_ylabel("Closing Prices")
plt.legend(loc="best")
ax.grid(True)
plt.show()


# In[83]:


rel_model, rel_fit_sarimax, rel_order, rel_seasonal_order = tsf.model_eval(rel_monthly_mean, arima=None)


# #### ARMA model

# In[84]:


#rel_arma_model, rel_arma_fitted = tsf.model_eval(rel_monthly_mean, arima=1, order=rel_order)


# In[85]:


#fc, se, conf = rel_arma_fitted.forecast(len(rel_test), alpha=0.05)


# In[86]:


# fc_series = pd.Series(fc, index=rel_test.index)
# lower_series = pd.Series(conf[:, 0], index=rel_test.index)
# upper_series = pd.Series(conf[:, 1], index=rel_test.index)
# plt.plot(rel_monthly_mean, label="Actual Stock Price")
# # plt.plot(rel_test, color="green", label="Actual Stock Price")
# plt.plot(fc_series, color="orange", label="Predicted Stock Price")
# plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=0.2)
# plt.xlabel("Date")
# plt.ylabel("Stock Price")
# plt.title("Reliance Stock Prediction using ARMA")
# plt.grid(True)
# plt.show()


# In[87]:


## Before 26-11-2009, split the stock price by 1/4
## Before 07-09-2017, split the stock price by 1/2


# #### SARIMAX model

# In[88]:


sarimax_rel_model_final = SARIMAX(rel_monthly_mean, order=rel_order, seasonal_order=rel_seasonal_order)
final_rel_result = sarimax_rel_model_final.fit()


# In[89]:


final_rel_result.plot_diagnostics(figsize=(10, 6))
plt.show()


# ##### Prediction

# In[90]:


# final_fc = final_rel_result.forecast(steps=len(rel_test))
# fc_df = pd.DataFrame(final_fc)
# fc_df.reset_index(inplace=True)
# fc_df.drop("index", axis=1, inplace=True)
# fc_df.set_index(rel_test.index, inplace=True)
# fc_df


# In[91]:


pred = final_rel_result.get_prediction(start=pd.to_datetime("2019-10-31"), dynamic=False)
pred_ci = pred.conf_int(alpha=0.05)
pred_df = pd.DataFrame(pred.predicted_mean)
pred_df


# In[92]:


plt.figure(figsize=(12, 5))
plt.grid(True)
plt.plot(rel_monthly_mean, label="Actual Stock Price")
plt.plot(pred_df["predicted_mean"], color="orange",label="One Step Ahead Forecast")
plt.fill_between(pred_ci.index,pred_ci["lower Close"], pred_ci["upper Close"], color="k", alpha=0.3)
plt.title("Reliance Stock Price Prediction")
plt.ylim(0, 3000)
plt.xlabel("Time")
plt.ylabel("Actual Stock Price")
plt.legend(loc="best")
plt.show()


# In[93]:


y_pred = pred.predicted_mean
y_true = rel_monthly_mean["Close"]
mse = ((y_pred - y_true) **2).mean()
print(f"The Mean Squared Error is {round(mse, 2)}")
print(f"The root Mean Squared Error is {round(np.sqrt(mse), 2)}")


# ##### One Step Ahead Forecast

# In[94]:


forecast = final_rel_result.get_forecast(steps=1000)
forecast_ci = forecast.conf_int()
# ax = rel_test.plot(label="Observed")
# forecast.predicted_mean.plot(ax=ax, label="Forecast")
# ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 0], color="k", alpha=0.25)
# ax.set_xlabel("Date")
# ax.set_ylabel("Stock Price")
# plt.legend()
# plt.show()
dates= pd.date_range(start="2021-10-04",periods=1000, freq="B")
forecast_series = pd.Series(forecast.predicted_mean)
forecast_df = pd.DataFrame(forecast_series)
forecast_df["Dates"] = dates
forecast_df.reset_index(inplace=True)
forecast_df.drop("index", axis=1, inplace=True)
forecast_df.set_index(forecast_df["Dates"], inplace=True)
forecast_df.drop("Dates", axis=1, inplace=True)

forecast_ci["Dates"] = dates
forecast_ci.set_index("Dates", inplace=True)
forecast_df["2023"]


# In[95]:


# list_1000_2500 = [i for i in range(1000, 2500, 250)]
req_dates = pd.date_range(start="04/01/2018", end="04/30/2022", freq="B").date
# b = "2021-07-01"
# a = "2021-05-01" 
plt.plot(forecast_df, color="orange", label="Forcasted Stock Price")
plt.plot(rel_monthly_mean, color="green", label="Train data")
# plt.axvspan(a, b, color="grey", alpha=0.2)
plt.fill_between(forecast_ci.index,forecast_ci["lower Close"], forecast_ci["upper Close"], color="k", alpha=0.2)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.ylim(0, 20000)
plt.title("Reliance Stock Price Forecasting")
plt.legend(loc="best")
plt.show()


# In[96]:


plt.figure(figsize=(12, 8))
plt.plot(forecast_df, label="Forecasted Stock Price")
plt.legend(loc="best")
plt.show()


# #### LSTM prediction

# In[97]:


rel_training_set = rel_monthly_mean[:int(len(rel_monthly_mean)*0.9)]
rel_test_set = rel_monthly_mean[int(len(rel_monthly_mean)*0.9):]


# In[98]:


sc= MinMaxScaler(feature_range=(0,1))
rel_training_set_scaled = sc.fit_transform(rel_training_set)


# In[99]:


X_train = []
y_train = []

for i in range(60, len(rel_training_set)):
    X_train.append(rel_training_set_scaled[i-60:i, 0])
    y_train.append(rel_training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train.shape


# In[100]:


model_lstm_rel = tsf.model_stock(X_train, y_train)


# In[101]:


model_lstm_rel.save_weights("LSTM_stocks_weights/")


# In[102]:


model_lstm_rel.save("complete_LSTM_stocks_model/")


# In[103]:


rel_LSTM_model = models.load_model("complete_LSTM_stocks_model/")


# In[104]:


rel_dataset_total = pd.concat([rel_training_set, rel_test_set], axis=0)

inputs = rel_dataset_total[len(rel_dataset_total) - len(rel_test_set) - 60:].values

inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_test.shape)


# In[105]:


predicted_stock_price_test = model_lstm_rel.predict(X_test)
predicted_stock_price_test = sc.inverse_transform(predicted_stock_price_test)


# In[106]:


predicted_stock_price_train = model_lstm_rel.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)


# In[107]:


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(rel_test_set.index,rel_test_set.values, color="red", label="Real Reliance Stock Price")
ax.plot(rel_test_set.index,predicted_stock_price_test, color="blue", label="Predicted Reliance Stock Price")
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.set_title("Reliance Stock Price Prediction")
ax.legend(loc="best")
plt.tight_layout()
plt.grid()
plt.show()


# In[108]:


rel_pred = pd.DataFrame(predicted_stock_price_test, index=rel_test_set.index)
rel_pred.columns = ["Predicted Closing Price"]


# In[109]:


rel_pred_v_actual = pd.concat([rel_test_set, rel_pred], axis=1)
rel_pred_v_actual


# ### HDFC BANK

# In[110]:


hdfc_bank = nifty_data.loc[nifty_data["Symbol"] == "HDFCBANK"]


# In[111]:


plt.figure(figsize=(20, 15))
sns.heatmap(hdfc_bank.corr(), annot=True, cmap="Blues")


# In[112]:


hdfc_bank_close = hdfc_bank[["Close"]]
hdfc_monthly_mean = hdfc_bank_close.resample("M").mean()
hdfc_monthly_mean
# hdfc_monthly_mean = hdfc_bank_close.resample("M").mean()
# hdfc_bank_close["2011-07-10": "2012-07-20"]
# hdfc_bank_close["2019-06": "2019-12"].value_counts().index.min()


# In[113]:


## Split 1 (14-07-2011) - 5:1 - Before the date split the price to 1/10
## Split 2 (19-11-2019) - 2:1
# hdfc_bank_close[:"2011-07-15"] = hdfc_bank_close[:"2011-07-15"] / 10
# hdfc_bank_close["2011-07-14":"2019-09-20"] = hdfc_bank_close["2011-07-14":"2019-09-20"] / 2


# #### Target Variable Evaluation

# In[114]:


tsf.stationarity_check(hdfc_monthly_mean)


# In[115]:


hdfc_seasonal_decomp = seasonal_decompose(hdfc_monthly_mean, model="multiplicative", freq=30)
fig = plt.figure()
fig = hdfc_seasonal_decomp.plot()
fig.set_size_inches(16, 9)


# In[116]:


tsf.plot_autocorrelation(hdfc_monthly_mean, verbose=1)


# In[117]:


rcParams["figure.figsize"] = 15, 6
hdfc_close_log = np.log(hdfc_monthly_mean)
hdfc_close_log_mean = hdfc_close_log.rolling(window=12).mean()
hdfc_close_log_std = hdfc_close_log.rolling(window=12).std()
plt.plot(hdfc_close_log_mean, color="red", label="log rolling mean")
plt.plot(hdfc_close_log_std, color="black", label="log rolling std")
plt.legend(loc="best")
plt.show()


# #### HDFC model

# In[118]:


hdfc_training_set = hdfc_close_log[:int(len(hdfc_close_log)*0.86)]
hdfc_test_set = hdfc_close_log[int(len(hdfc_close_log)*0.86):]
fig, ax = plt.subplots(figsize=(12, 6))
ax.grid(True)
ax.plot(hdfc_training_set.index, np.exp(hdfc_training_set), color="blue", label="training set")
ax.plot(hdfc_test_set.index, np.exp(hdfc_test_set), color="green", label="test set")
ax.set_xlabel("Dates")
ax.set_ylabel("Stock Price")
ax.legend(loc="best")
plt.show()


# In[119]:


hdfc_model, hdfc_fit_sarimax, hdfc_order, hdfc_seasonal_order = tsf.model_eval(hdfc_monthly_mean, arima=None)


# #### ARMA model

# In[120]:


# arma_hdfc_model, arma_hdfc_fit = tsf.model_eval(hdfc_training_set, arima=1, order=hdfc_order)


# In[121]:


# fc, se, conf = arma_hdfc_fit.forecast(steps=len(hdfc_test_set), alpha=0.05)


# In[122]:


# fc_series = pd.Series(fc, index=hdfc_test_set.index)
# # upper_series = pd.Series(conf[:, 1], index=hdfc_test_set.index)
# lower_series = pd.Series(conf[:, 0], index=hdfc_test_set.index)
# plt.figure(figsize=(12, 8))
# # plt.plot(np.exp(hdfc_training_set), label="Train data")
# plt.plot(hdfc_monthly_mean,color="green", label="Actual Stock Price")
# plt.plot(fc_series, color="orange", label="Predicted Stock Price")
# plt.fill_between(lower_series.index, lower_series, np.exp(upper_series), alpha=0.2, color="k")
# plt.xlabel("Date")
# plt.ylabel("Stock Price")
# plt.title("HDFC Bank Stock Price prediction")
# plt.legend(loc="best")
# plt.show()


# #### SARIMAX model

# In[123]:


final_hdfc_model = SARIMAX(hdfc_monthly_mean, order=hdfc_order, seasonal_order=hdfc_seasonal_order)
final_hdfc_fit = final_hdfc_model.fit()


# In[124]:


final_hdfc_fit.plot_diagnostics(figsize=(15, 8))
plt.show()


# ##### Prediction

# In[153]:


hdfc_pred = final_hdfc_fit.get_prediction(start = pd.to_datetime("2019-10-31"), dynamic=False)
pred_ci = hdfc_pred.conf_int()
pred_ci
hdfc_pred_series=pd.Series(hdfc_pred.predicted_mean)


# In[154]:


plt.figure(figsize=(15, 8))
plt.grid(True)
# plt.plot(np.exp(hdfc_training_set), color="blue", label="Training Set")
plt.plot(hdfc_monthly_mean, color="green", label="Actual Stock Price")
plt.plot(hdfc_pred_series, color="orange", label="One Step Ahead Forecast")
plt.fill_between(pred_ci.index, pred_ci["lower Close"], pred_ci["upper Close"], alpha=0.2, color="k")
# plt.yticks([x for x in range(0, 2500, 250)])
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("HDFC Stock Price Prediction")
plt.legend(loc="best")
plt.show()


# ##### One Step Ahead Forecast

# In[127]:


hdfc_forecast = final_hdfc_fit.get_forecast(steps=1000)
hdfc_forecast_ci = hdfc_forecast.conf_int()

dates= pd.date_range(start="2019-10-04", periods=1000, freq="B")
hdfc_forecast_series = pd.Series(hdfc_forecast.predicted_mean)
hdfc_forecast_series
hdfc_forecast_df = pd.DataFrame(hdfc_forecast_series)
hdfc_forecast_df.reset_index(inplace=True)
hdfc_forecast_df["Dates"] = dates
hdfc_forecast_df.set_index("Dates", inplace=True)
# hdfc_forecast_df.drop("Dates",axis=1, inplace=True)
# hdfc_forecast_df
hdfc_forecast_df.drop("index", axis=1, inplace=True)

hdfc_forecast_ci["Dates"] = dates
hdfc_forecast_ci.set_index("Dates", inplace=True)


# In[128]:


plt.figure(figsize=(12, 8))
a = "2021-05-01"
b = "2021-07-01"
plt.grid(True)
plt.plot(hdfc_monthly_mean, color="green", label="Actual Stock Price")
plt.plot(hdfc_forecast_df, color="orange", label="Forecasted Stock Price")
# plt.axvspan(a, b, color="grey", alpha=0.2)
plt.fill_between(hdfc_forecast_ci.index, hdfc_forecast_ci["lower Close"], hdfc_forecast_ci["upper Close"], alpha=0.2, color="k")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("HDFC Stock Price Forecast")
plt.legend(loc="best")
plt.show()


# In[129]:


plt.figure(figsize=(12, 7))
plt.plot(hdfc_forecast_df, label="Forecasted Stock Price")
plt.title("Forecast")
plt.legend()
plt.show()


# #### LSTM prediction

# In[130]:


hdfc_train = hdfc_monthly_mean[:int(len(hdfc_monthly_mean)*0.9)]
hdfc_test = hdfc_monthly_mean[int(len(hdfc_monthly_mean)*0.9):]
sc = MinMaxScaler(feature_range=(0, 1))
hdfc_train_scaled = sc.fit_transform(hdfc_train)


# In[131]:


X_train = []
y_train = []
for i in range(60, len(hdfc_train_scaled)):
    X_train.append(hdfc_train_scaled[i-60:i, 0])
    y_train.append(hdfc_train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train.shape


# In[132]:


hdfc_model = tsf.model_stock(X_train, y_train)


# In[133]:


hdfc_dataset_total = pd.concat([hdfc_train, hdfc_test], axis=0)

inputs = hdfc_dataset_total[len(hdfc_dataset_total) - len(hdfc_test) - 60:].values

inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_test.shape)


# In[134]:


hdfc_pred = hdfc_model.predict(X_test)
hdfc_pred = sc.inverse_transform(hdfc_pred)


# In[135]:


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hdfc_test.index,hdfc_test.values, color="red", label="Real Reliance Stock Price")
ax.plot(hdfc_test.index,hdfc_pred, color="blue", label="Predicted Reliance Stock Price")
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.set_title("Reliance Stock Price Prediction")
ax.legend(loc="best")
plt.tight_layout()
plt.grid()
plt.show()


# ## Global function

# In[161]:


def stock_market_prediction(df, symbol, heatmap=None,stationarity=None, decompose=None, ac=None, log_plot=None, best_params=None, arma_model=None, sarimax=None, lstm=None, test=None, date=True):
    stock_df= df.loc[df["Symbol"] == symbol]
    '''
    
    '''
#     stock_df["Date"] = pd.to_datetime(stock_df["Date"])
#     stock_df.set_index(stock_df["Date"], inplace=True)
#     stock_df.drop("Date", axis=1, inplace=True)

    #Plot the heatmap of the dataset
    if heatmap != None:
        plt.figure(figsize=(20, 15))
        sns.heatmap(stock_df.corr(), annot=True, cmap="Blues")
        plt.show()


    ts = stock_df[["Close"]]
    ts_monthly_mean = ts.resample("M").mean()
    ts_monthly_mean.fillna(method="bfill", inplace=True)
    
    if stationarity != None:
        tsf.stationarity_check(ts_monthly_mean)

    if decompose != None:
        if len(ts_monthly_mean) >= 60:
            ts_decomp = seasonal_decompose(ts_monthly_mean, model="multiplicative", freq=30)
            fig = plt.figure()
            fig = ts_decomp.plot()
            fig.set_size_inches(16, 9)
        else:
            ts_weekly_mean = ts.resample("W").mean()
            ts_weekly_mean.fillna(method="bfill", inplace=True)
            ts_decomp = seasonal_decompose(ts_weekly_mean, model="multiplicative", freq=30)
            fig = plt.figure()
            fig = ts_decomp.plot()
            fig.set_size_inches(16, 9)
            
        

    if ac != None:
        if len(ts_monthly_mean) >=60:
            tsf.plot_autocorrelation(ts_monthly_mean, verbose=1)
        else:
            tsf.plot_autocorrelation(ts_weekly_mean, verbose=1)
    
    if log_plot != None:
        plt.figure(figsize=(15, 7))
        ts_log = np.log(ts_monthly_mean)
        moving_avg = ts_log.rolling(12).mean()
        std= ts_log.rolling(12).std()
        plt.plot(moving_avg, color="red", label="Mean")
        plt.plot(std, color="black", label="Standard Deviation")
        plt.title("Moving Average")
        plt.legend(loc="best")
        plt.show()


# ####  Model



    ts_train = ts_log[:int(len(ts_log) *0.8)]
    ts_test = ts_log[int(len(ts_log) *0.8):]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(ts_test, color="green", label="Actual Stock Price")
    ax.plot(ts_train, label="train data")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Closing Prices")
    plt.legend(loc="best")
    ax.grid(True)
    plt.show()

    if best_params != None:
        ts_model, ts_fit_sarimax, order, seasonal_order= tsf.model_eval(ts_monthly_mean, arima=None)
        
#     if arma_model != None:
#         ts_arma_model, ts_arma_fitted = tsf.model_eval(ts_monthly_mean, arima=1, order= (1,1))
#         fc, se, conf = ts_arma_fitted.forecast(len(ts_test), alpha=0.05)


#     fc_series = pd.Series(fc, index=ts_monthly_mean.index)
#     lower_series = pd.Series(conf[:, 0], index=ts_monthly_mean.index)
#     upper_series = pd.Series(conf[:, 1], index=ts_monthly_mean.index)
#     plt.plot(ts_monthly_mean, label="train data")
# #     plt.plot(np.exp(ts_test), color="green", label="Actual Stock Price")
#     plt.plot(fc_series, color="orange", label="Predicted Stock Price")
#     plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=0.2)
#     plt.xlabel("Date")
#     plt.ylabel("Stock Price")
#     plt.title("Stock Prediction using ARMA")
#     plt.show()


# #### SARIMAX model

    if sarimax != None:
        
        sarimax_model_final = SARIMAX(ts_monthly_mean, order=order, seasonal_order=seasonal_order)
        final_result = sarimax_model_final.fit()
        

    final_result.plot_diagnostics(figsize=(10, 6))
    plt.show()


# ##### Prediction


# final_fc = final_rel_result.forecast(steps=len(rel_test))
# fc_df = pd.DataFrame(final_fc)
# fc_df.reset_index(inplace=True)
# fc_df.drop("index", axis=1, inplace=True)
# fc_df.set_index(rel_test.index, inplace=True)
# fc_df

    if date:
        pred = final_result.get_prediction(start=pd.to_datetime("2019-10-31"), dynamic=False)
        pred_ci = pred.conf_int(alpha=0.05)
    else:
        pred = final_result.get_prediction(start=pd.to_datetime("2020-04-01"), dynamic=False)
        pred_ci = pred.conf_int(alpha=0.05)



    plt.figure(figsize=(12, 5))
    plt.grid(True)
    plt.plot(ts_monthly_mean, label="Actual Stock Price")
#     plt.plot(np.exp(ts_test), color="green", label="Test Data")
    plt.plot(pred.predicted_mean, label="Predicted Stock Price")
    plt.fill_between(pred_ci.index, pred_ci["lower Close"], pred_ci["upper Close"], color="k", alpha=0.3)
    plt.title(f"{symbol} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Actual Stock Price")
    plt.legend(loc="best")
    plt.show()




    y_pred = pred.predicted_mean
    y_true = ts_monthly_mean["Close"]
    mse = ((y_pred - y_true) **2).mean()
    print(f"The Mean Squared Error is {round(mse, 2)}")
    print(f"The root Mean Squared Error is {round(np.sqrt(mse), 2)}")


# ##### One Step Ahead Forecast



    forecast = final_result.get_forecast(steps=1000)
    pred_ci = forecast.conf_int()
# ax = rel_test.plot(label="Observed")
# forecast.predicted_mean.plot(ax=ax, label="Forecast")
# ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 0], color="k", alpha=0.25)
# ax.set_xlabel("Date")
# ax.set_ylabel("Stock Price")
# plt.legend()
# plt.show()
    dates= pd.date_range(start="2019-10-31", periods= 1000, freq="B")
    forecast_series = pd.Series(forecast.predicted_mean)
    forecast_df = pd.DataFrame(forecast_series)
    forecast_df["Dates"] = dates
    forecast_df.reset_index(inplace=True)
    forecast_df.drop("index", axis=1, inplace=True)
    forecast_df.set_index(forecast_df["Dates"], inplace=True)
    forecast_df.drop("Dates",axis=1, inplace=True)

    pred_ci["Dates"] = dates
    pred_ci.set_index("Dates", inplace=True)


# list_1000_2500 = [i for i in range(1000, 2500, 250)]
    req_dates = pd.date_range(start="04/01/2018", end="04/30/2022", freq="B").date
# b = "2021-07-01"
# a = "2021-05-01" 
    plt.plot(forecast_df, color="orange", label="Forcasted Stock Price")
    plt.plot(ts_monthly_mean, color="green", label="Actual Stock Price")
#     plt.plot(np.exp(ts_test), color="blue", label="Actual Stock Price")
    # plt.axvspan(a, b, color="grey", alpha=0.2)
    plt.fill_between(pred_ci.index,pred_ci["lower Close"], pred_ci["upper Close"], alpha=0.2, color="k")
# plt.yticks(list_1000_2500)
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend(loc="best")
    plt.title(f"{symbol} Forecast")
    plt.show()




    plt.figure(figsize=(12, 8))
    plt.plot(forecast_df, label="Forecasted Stock Price")
    plt.legend(loc="best")
    plt.show()


# #### LSTM prediction

    if lstm != None:
        if len(ts_monthly_mean) >= 60:
            ts_training_set = ts_monthly_mean[:int(len(ts_monthly_mean)*0.9)]
            ts_test_set = ts_monthly_mean[int(len(ts_monthly_mean)*0.9):]
        else:
            ts_training_set = ts_monthly_mean[:int(len(ts_monthly_mean)*0.9)]
            ts_test_set = ts_monthly_mean[int(len(ts_monthly_mean)*0.9):]


        sc= MinMaxScaler(feature_range=(0,1))
        ts_training_set_scaled = sc.fit_transform(ts_training_set)


        X_train = []
        y_train = []
        for i in range(60, len(ts_training_set_scaled)):
            X_train.append(ts_training_set_scaled[i-60:i, 0])
            y_train.append(ts_training_set_scaled[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        print(X_train.shape)



        model_lstm_ts = tsf.model_stock(X_train, y_train)

        if test != None:
            X_test = []
            ts_dataset_total = pd.concat([ts_training_set, ts_test_set], axis=0)

            inputs = ts_dataset_total[len(ts_dataset_total) - len(ts_test_set) - 60:].values

            inputs = inputs.reshape(-1, 1)
            inputs = sc.transform(inputs)

            X_test = []
            for i in range(60, len(inputs)):
                X_test.append(inputs[i-60:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                            
    


            predicted_stock_price_test = model_lstm_ts.predict(X_test)
            predicted_stock_price_test = sc.inverse_transform(predicted_stock_price_test)




    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts_test_set.index,ts_test_set, color="red", label=f"Real {symbol} Stock Price")
    ax.plot(ts_test_set.index,predicted_stock_price_test, color="blue", label=f"Predicted {symbol} Stock Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"{symbol} Stock Price Prediction")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
    return


# ### Indian Stocks

# #### ADANIPORTS

# In[137]:


stock_market_prediction(nifty_data,symbol="ADANIPORTS",heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1,sarimax=1, lstm=1, test=1)


# #### POWERGRID

# In[138]:


stock_market_prediction(nifty_data, symbol="POWERGRID", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### ICICI BANK

# In[139]:


stock_market_prediction(nifty_data, symbol="ICICIBANK", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### KOTAK BANK

# In[140]:


stock_market_prediction(nifty_data, symbol="KOTAKBANK", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### ASTRAL

# In[141]:


stock_market_prediction(NSE_data, symbol="ASTRAL", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### SUMCHEM

# In[142]:


stock_market_prediction(NSE_data, symbol="SUMCHEM", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### BERGEPAINT

# In[143]:


stock_market_prediction(NSE_data, symbol="BERGEPAINT", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### IGL

# In[144]:


stock_market_prediction(NSE_data, symbol="IGL", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### KANSAINER
# 

# In[145]:


stock_market_prediction(NSE_data, symbol="KANSAINER", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### RELAXO

# In[146]:


stock_market_prediction(NSE_data, symbol="RELAXO", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### CDSL

# In[206]:


#stock_market_prediction(NSE_data, symbol="CDSL", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### HDFCLIFE

# In[205]:


# stock_market_prediction(NSE_data, symbol="HDFCLIFE", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### IEX

# In[170]:


# stock_market_prediction(NSE_data, symbol="IEX", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### SIEMENS

# In[165]:


stock_market_prediction(NSE_data, symbol="SIEMENS", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### CAMS

# In[204]:


# stock_market_prediction(NSE_data, symbol="CAMS", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### PRINCEPIPE

# In[202]:


# stock_market_prediction(NSE_data, symbol="PRINCEPIPE", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1, date=False)


# #### SBICARD

# In[203]:


# stock_market_prediction(NSE_data, symbol="SBICARD", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### ROUTE

# In[171]:


# stock_market_prediction(NSE_data, symbol="ROUTE", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# ### American Stocks

# #### AAPL

# In[197]:


stock_market_prediction(faang, symbol="AAPL", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### AMZN

# In[198]:


stock_market_prediction(faang, symbol="AMZN", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### FB

# In[199]:


stock_market_prediction(faang, symbol="FB", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### GOOG

# In[200]:


stock_market_prediction(faang, symbol="GOOG", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)


# #### NFLX

# In[201]:


stock_market_prediction(faang, symbol="NFLX", heatmap=1, stationarity=1, decompose=1, ac=1, log_plot=1, best_params=1, arma_model=1, sarimax=1, lstm=1, test=1)

