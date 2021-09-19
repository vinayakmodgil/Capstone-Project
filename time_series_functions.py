import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from IPython.display import display
import seaborn as sns


def stationarity_check(ts, window=12, plot=True, index=["Ad Fuller Result"]):
    '''
    Adapted from github, with tweaks to the function. 
    
    This function performs/displays dickey fuller test and
    calculates/plots the rolling mean and standard deviation
    of a time series data.
    
    stationarity_check:
    ts -> time series dataframe
    
    '''
    
    #Check the rolling statistics
    roll_mean = ts.rolling(window=window, center=False).mean()
    roll_std = ts.rolling(window=window, center=False).std()
    
    #Perform Dickey Fuller Test
    df = tsa.stattools.adfuller(ts)
    
    #Extract the useful columns 
    names = ["Test Statistic", "p-value", "#Lags Used", "No of Obs"]
    
    #Convert the useful columns to dict
    test = dict(zip(names, df[:4]))
    
    test["p<0.05"] = test["p-value"] < 0.05
    test["Stationary?"] = test["p<0.05"]
    
    req_df = pd.DataFrame(test, index=index)
    
    #Plot the dataset and the time series
    if plot:
        fig = plt.figure(figsize=(20, 12))
        plt.plot(ts, color="blue")
        plt.plot(roll_mean, color="red", label=f"Rolling Mean (Window={window})")
        plt.plot(roll_std, color="black", label=f"Rolling Std (Window={window})")
        plt.legend()
        plt.title("Rolling Mean & Standard Deviation")
        display(req_df)
        plt.show()
    
    return 
                 
def plot_autocorrelation(ts, plot=True, period=13, verbose=0):
    
    if verbose != 0:
  
        period_list = [x for x in range(1, period)]
        fig, ax = plt.subplots(nrows=len(period_list), figsize=(50, 40))
    
        for i, p in enumerate(period_list):
            if plot:
                ax[i].plot(pd.concat([ts, ts.shift(periods=p)], axis=1), label=f"period: {p}")
                ax[i].set_title(f"Normal vs Shift(period {p})")
            
            
        print("***"*20)
        print("AUTOCORRELATION and PARTIAL AUTOCORRELATION")
        print("***"*20)    
    
        plt.figure(figsize=(12, 6))
        plot_acf(ts, lags= 50, title="Autocorrelation Plot")
        plt.figure(figsize=(12, 6))
        plot_pacf(ts, lags=50, title="Partial Autocorrelation Plot")
    
    return
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

