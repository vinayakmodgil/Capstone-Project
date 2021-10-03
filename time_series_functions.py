import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from IPython.display import display
import seaborn as sns
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from matplotlib.pylab import rcParams

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential


def stationarity_check(ts, window=12, plot=True, index=["Ad Fuller Result"], center=False, ax=0):
    '''
    Adapted from github, with tweaks to the function. 
    
    This function performs/displays dickey fuller test and
    calculates/plots the rolling mean and standard deviation
    of a time series data.
    
    stationarity_check:
    ts -> time series dataframe
    
    '''
    ts_diff = ts.diff().dropna()
    
    #Check the rolling statistics
    roll_mean = ts.rolling(window=window, center=center).mean()
    roll_std = ts.rolling(window=window, center=center).std()
    
    #Perform Dickey Fuller Test
    df = tsa.stattools.adfuller(ts, autolag="AIC")
    
    #Extract the useful columns 
    names = ["Test Statistic", "p-value", "#Lags Used", "No of Obs"]
    
    #Convert the useful columns to dict
    test = dict(zip(names, df[:4]))
    
    test["p<0.05"] = test["p-value"] < 0.05
    test["Stationary?"] = test["p<0.05"]
    
    req_df = pd.DataFrame(test, index=index)
    
    #Plot the dataset and the time series
    if plot:
        rcParams["figure.figsize"] = 15, 7
        ax = roll_mean.plot(color="red", label=f"Rolling Mean (Window={window})")
        roll_std.plot(color="black",ax=ax, label=f"Rolling Std (Window={window})")
        ts.plot(color="yellow",ax=ax, label="original")
        plt.legend(loc="best")
        plt.title("Rolling Mean & Standard Deviation")
        display(req_df)
        plt.show()
    
    return 
                 
def plot_autocorrelation(ts, plot=True, period=13, verbose=0):
    '''
    
    '''
    
    if verbose != 0:
        ts_diff = ts.diff().dropna()
        period_list = [x for x in range(1, period)]
        fig, ax = plt.subplots(nrows=len(period_list), figsize=(50, 40))
    
        for i, p in enumerate(period_list):
            if plot:
                ax[i].plot(pd.concat([ts.shift(periods=p), ts], axis=1), label=f"period: {p}")
                ax[i].set_title(f"Normal vs Shift(period {p})")
            
            
        print("***"*20)
        print("AUTOCORRELATION and PARTIAL AUTOCORRELATION")
        print("***"*20)    
    
        plt.figure(figsize=(15, 7))
        plot_acf(ts, lags= 24, title="Autocorrelation Plot")
        
        plt.figure(figsize=(15, 7))
        plot_pacf(ts, lags=24, title="Partial Autocorrelation Plot")
        plt.tight_layout()
    
    return
        
        
def model_eval(ts, prev_ts=None, arima=None, order=(1, 0)):
    '''
    part of the code is adapted from github, tweaked into a presentable function
    '''
    #Previous ARIMA model
    if prev_ts != None:
        model_prev_arma = ARMA(prev_ts, order=order)
        model_prev_res = model_prev_arma.fit()
        print(model_prev_res.summary())
        print("\n\n")
        print("***" * 30)
        print("Parameters")
        print("***" * 30)
        print("\n\n")
        print(model_prev_res.params)
        
        return model_prev_arma, model_prev_res
        
    #ARIMA model
    elif arima != None:
        model_arma = ARMA(ts, order=order)
        model_res = model_arma.fit()
        print(model_res.summary())
        print("\n\n")
        print("***" * 30)
        print("Parameters")
        print("***" * 30)
        print("\n\n")
        print(model_res.params)    
        
        return model_arma, model_res
    
    else:
        # SARIMAX model
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        
        pdqs = [(x[0], x[1], x[2], 12) for x in pdq]
        
        ans = []
        
        for comb in pdq:
            for combs in pdqs:
                model_sarimax = SARIMAX(ts, order=comb, seasonal_order=combs, enforce_stationarity=False, enforce_invertibility=False)
            
                fit_sarimax = model_sarimax.fit()
                ans.append([comb, combs, fit_sarimax.aic])
                print(f"ARIMA {comb} x {combs}:AIC calculated = {fit_sarimax.aic}")
        
        print("\n\n")
        ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
        print(ans_df.loc[ans_df['aic'].idxmin()])
        
        return model_sarimax, fit_sarimax, ans_df["pdq"].loc[ans_df["aic"].idxmin()], ans_df["pdqs"].loc[ans_df["aic"].idxmin()]


def model_stock(X_train, y_train):
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.02))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.02))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.02))
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.02))
    
    model.add(Dense(units=1))
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    return model    
            
        
    
    
    
    
    
    
    
    
    
    
    
    

