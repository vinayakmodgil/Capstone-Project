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


def stock_market_prediction(df, symbol, heatmap=None,stationarity=None, decompose=None, ac=None, log_plot=None, best_params=None, arma_model=None, sarimax=None):
    stock_df= df.loc[df["Symbol"] == symbol]

    #Plot the heatmap of the dataset
    if heatmap != None:
        plt.figure(figsize=(20, 15))
        sns.heatmap(df.corr(), annot=True, cmap="Blues")
        plt.show()


    ts = stock_df[["Close"]]
    
    if stationarity != None:
        tsf.stationarity_check(ts)

    if decompose != None:
        ts_decomp = seasonal_decompose(ts, model="multiplicative", freq=30)
        fig = plt.figure()
        fig = ts_decomp.plot()
        fig.set_size_inches(16, 9)

    if ac != None:
        tsf.plot_autocorrelation(rel_close, verbose=1)
    
    if log_plot != None:
        rcParams["figure.figsize"] = 15, 6
        ts_log = np.log(ts)
        moving_avg = ts_log.rolling(12).mean()
        std= ts_log.rolling(12).std()
        plt.plot(moving_avg, color="red", label="Mean")
        plt.plot(std, color="black", label="Standard Deviation")
        plt.title("Moving Average")
        plt.legend(loc="best")
        plt.show()


# #### Reliance Model

# In[33]:


    ts_train = ts_log[:int(len(rel_close_log) *0.8)]
    ts_test = ts_log[int(len(rel_close_log) *0.8):]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(ts_test, color="green", label="Actual Stock Price test")
    ax.plot(ts_train, label="Actual Stock Price train")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Closing Prices")
    plt.legend(loc="best")
    ax.grid(True)
    plt.show()

    if best_params != None:
        ts_model, ts_fit_sarimax, order, seasonal_order= tsf.model_eval(ts_train, arima=None)
        
    if arma_model != None:
        ts_arma_model, ts_arma_fitted = tsf.model_eval(ts_train, arima=1, order=order)


    fc, se, conf = ts_arma_fitted.forecast(len(ts_test), alpha=0.05)


    fc_series = pd.Series(fc, index=rel_test.index)
    lower_series = pd.Series(conf[:, 0], index=rel_test.index)
    upper_series = pd.Series(conf[:, 1], index=rel_test.index)
    plt.plot(np.exp(rel_train), label="train data")
    plt.plot(np.exp(rel_test), color="green", label="Actual Stock Price")
    plt.plot(np.exp(fc_series), color="orange", label="Predicted Stock Price")
    plt.fill_between(lower_series.index, np.exp(lower_series), np.exp(upper_series), color="k", alpha=0.2)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Reliance Stock Prediction using ARMA")
    plt.show()


# #### SARIMAX model

    if sarimax != None:

        sarimax_model_final = SARIMAX(rel_train, order=order, seasonal_order=seasonal_order)
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


# In[41]:


    pred = final_result.get_prediction(start=pd.to_datetime(ts.index[0]), dynamic=False)
    pred_ci = pred.conf_int(alpha=0.05)
    pred.predicted_mean
    pred_ci["Dates"] = ts_train.index
    pred_ci.set_index("Dates", inplace=True)
    pred_ci


# In[42]:


    plt.figure(figsize=(12, 5))
    plt.grid(True)
    plt.plot(np.exp(ts_train), label="Actual Stock Price")
    plt.plot(np.exp(ts_test), color="green", label="Test Data")
    plt.plot(np.exp(pred.predicted_mean), label="Predicted Stock Price")
    plt.fill_between(pred_ci.index, np.exp(pred_ci["lower Close"]), np.exp(pred_ci["upper Close"]), color="k", alpha=0.3)
    plt.title(f"{symbol} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Actual Stock Price")
    plt.legend(loc="best")
    plt.show()


# In[43]:


    y_pred = pred.predicted_mean
    y_true = ts_train["Close"]
    mse = ((y_pred - y_true) **2).mean()
    print(f"The Mean Squared Error is {round(mse, 2)}")
    print(f"The root Mean Squared Error is {round(np.sqrt(mse), 2)}")


# ##### One Step Ahead Forecast

# In[ ]:


    forecast = final_rel_result.get_forecast(steps=365)
    pred_ci = forecast.conf_int()
# ax = rel_test.plot(label="Observed")
# forecast.predicted_mean.plot(ax=ax, label="Forecast")
# ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 0], color="k", alpha=0.25)
# ax.set_xlabel("Date")
# ax.set_ylabel("Stock Price")
# plt.legend()
# plt.show()
    dates= pd.date_range(start=ts_train.index[-1], periods= 1500, freq="B")
    forecast_series = pd.Series(forecast.predicted_mean)
    forecast_df = pd.DataFrame(forecast_series)
    forecast_df["Dates"] = dates
    forecast_df.reset_index(inplace=True)
    forecast_df.drop("index", axis=1, inplace=True)
    forecast_df.set_index(forecast_df["Dates"], inplace=True)
    forecast_df.drop("Dates",axis=1, inplace=True)

    pred_ci["Dates"] = dates
    pred_ci.set_index("Dates", inplace=True)


# In[ ]:


# list_1000_2500 = [i for i in range(1000, 2500, 250)]
    req_dates = pd.date_range(start="04/01/2018", end="04/30/2022", freq="B").date
# b = "2021-07-01"
# a = "2021-05-01" 
    plt.plot(np.exp(forecast_df), color="orange", label="Forcasted Stock Price")
    plt.plot(np.exp(ts_train), color="green", label="Actual Stock Price")
    plt.plot(np.exp(ts_test), color="blue", label="test data")
    # plt.axvspan(a, b, color="grey", alpha=0.2)
    plt.fill_between(pred_ci.index,np.exp(pred_ci["lower Close"]), np.exp(pred_ci["upper Close"]), alpha=0.2, color="k")
# plt.yticks(list_1000_2500)
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend(loc="best")
    plt.show()


# In[46]:


    plt.figure(figsize=(12, 8))
    plt.plot(np.exp(forecast_df), label="Forecasted Stock Price")
    plt.legend(loc="best")
    plt.show()


# #### LSTM prediction

# In[47]:


    ts_training_set = ts[:int(len(rel_close)*0.9)]
    ts_test_set = ts[int(len(rel_close)*0.9):]


# In[48]:


    sc= MinMaxScaler(feature_range=(0,1))
    ts_training_set_scaled = sc.fit_transform(ts_training_set)


# In[49]:


    X_train = []
    y_train = []

    for i in range(60, len(ts_training_set)):
        X_train.append(ts_training_set_scaled[i-60:i, 0])
        y_train.append(ts_training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        X_train.shape


# In[50]:


    model_lstm_rel = tsf.model_stock(X_train, y_train)


    ts_dataset_total = pd.concat([ts_training_set, ts_test_set], axis=0)

    inputs = ts_dataset_total[len(ts_dataset_total) - len(ts_test_set) - 60:].values

    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    


# In[55]:


    predicted_stock_price_test = model_lstm_ts.predict(X_test)
    predicted_stock_price_test = sc.inverse_transform(predicted_stock_price_test)


# In[57]:


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rel_test_set.index,rel_test_set.values, color="red", label="Real Reliance Stock Price")
    ax.plot(rel_test_set.index,predicted_stock_price_test, color="blue", label=f"Predicted {symbol} Stock Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_title("Reliance Stock Price Prediction")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


# In[58]:


# rel_pred_train = pd.DataFrame(predicted_stock_price_train)
# rel_pred_test = pd.DataFrame(predicted_stock_price_test)
# rel_pred_train.set_index(rel_dataset_train.index)
# rel_pred_test.set_index(rel_dataset_test.index)


# In[59]:


#rel_pred = pd.DataFrame(predicted_stock_price_test, index=rel_test_set.index)
#rel_pred.columns = ["Predicted Closing Price"]


# In[60]:


#rel_pred_v_actual = pd.concat([rel_test_set, rel_pred], axis=1)
#rel_pred_v_actual
