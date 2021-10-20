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


def stock_market_prediction(df, symbol, heatmap=None,stationarity=None, decompose=None, ac=None, log_plot=None, best_params=None, arma_model=None, sarimax=None, lstm=None, test=None, date=True):
    stock_df= df.loc[df["Symbol"] == symbol]
    '''
    A function that uses all parts of the predictions 
    and data analysis and prints the outputs accordingly.
    
    stock_market_prediction :
    df -> a dataframe
    symbol -> the symbol column in the df.
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
    
    ts_monthly_mean_diff = ts_monthly_mean.diff(periods=1)
    
    if stationarity != None:
        tsf.stationarity_check(ts_monthly_mean_diff.dropna())

    if decompose != None:
        ts_decomp = seasonal_decompose(ts_monthly_mean_diff.dropna(), model="additive", freq=30)
        fig = plt.figure()
        fig = ts_decomp.plot()
        fig.set_size_inches(16, 9)
#         else:
#             ts_weekly_mean = ts.resample("W").mean()
#             ts_weekly_mean.fillna(method="bfill", inplace=True)
#             ts_decomp = seasonal_decompose(ts_weekly_mean, model="multiplicative", freq=30)
#             fig = plt.figure()
#             fig = ts_decomp.plot()
#             fig.set_size_inches(16, 9)
            
        

    if ac != None:
        tsf.plot_autocorrelation(ts_monthly_mean_diff.dropna(), verbose=1)
#         else:
#             tsf.plot_autocorrelation(ts_weekly_mean_diff.dropna(), verbose=1)
    
    if log_plot != None:
        plt.figure(figsize=(15, 7))
        ts_log = np.log(ts_monthly_mean).diff().dropna()
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
        ts_model, ts_fit_sarimax, order, seasonal_order= tsf.model_eval(ts_monthly_mean_diff.dropna(), arima=None)
        
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
        
        sarimax_model_final = SARIMAX(ts_monthly_mean_diff.dropna(), order=order, seasonal_order=seasonal_order)
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
    plt.plot(ts_monthly_mean_diff.dropna(), label="Actual Stock Price")
#     plt.plot(np.exp(ts_test), color="green", label="Test Data")
    plt.plot(pred.predicted_mean, label="Predicted Stock Price")
    plt.fill_between(pred_ci.index, pred_ci["lower Close"], pred_ci["upper Close"], color="k", alpha=0.3)
    plt.title(f"{symbol} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Actual Stock Price")
    plt.legend(loc="best")
    plt.show()




    y_pred = pred.predicted_mean
    y_true = ts_monthly_mean_diff.dropna()["Close"]
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
    dates= pd.date_range(start="2021-10-04", periods= 1000, freq="B")
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
    plt.plot(ts_monthly_mean_diff.dropna(), color="green", label="Actual Stock Price")
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
