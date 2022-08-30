import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import datetime
import random
import matplotlib.pyplot as plt
# %matplotlib inline

import sklearn
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.stats import random_correlation
import scipy.optimize as optimize
import scipy.interpolate as sci
import time
import tqdm

import import_ipynb

import tensorflow as tf
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials, space_eval
from hyperopt import rand
import sys
import math

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import itertools

    
def sharpe(rets):
    total_ret = tf.reduce_prod(1 + rets) - 1
    return float(total_ret/np.std(rets))

def sortino(rets):
    neg_rets = rets[rets<0]
    total_ret = tf.reduce_prod(1 + rets) - 1
    return float(total_ret / np.std(neg_rets))

class Data:
    '''
    Parameters:

    '''
    def __init__(self, index_list, start_date, end_date, n_past, mvo_shift):
        self.index_list = index_list
        self.size = len(index_list)
        self.start_date = start_date
        self.end_date = end_date
        self.cov = None
        self.pi_matrix = None
        self.prices = []
        self.returns = None
        self.n_past = n_past
        self.shift = mvo_shift
        self.shift_returns = None
        self.shift_returns_mean = None
        self.shift_returns_var = None

    def get_shift_returns(self):
        self.shift_returns = self.prices/self.prices.shift(self.shift) - 1
        return self.shift_returns

    def compute_cov(self):    
        """to-do: """
        # np.random.seed(self.seed)
        # eigen = np.random.dirichlet(np.ones(self.size), size=1)[0]*self.size
        # C = random_correlation.rvs(eigen, random_state=self.seed)
        # D = np.diag(np.random.uniform(low=self.sd_low, high=self.sd_high, size=(1, self.size))[0])
        
        # self.generate_df()
        StockList = self.index_list
        filter_len = self.shift
        self.shift_returns = self.get_shift_returns()
        NumStocks = len(StockList)
        covariance = pd.DataFrame()
        for FirstStock in np.arange(NumStocks-1):
            for SecondStock in np.arange(FirstStock+1,NumStocks):
                ColumnTitle = StockList[FirstStock] + '-' + StockList[SecondStock]
                covariance[ColumnTitle] = self.shift_returns[StockList[FirstStock]].ewm(span=filter_len).cov(self.shift_returns[StockList[SecondStock]])
        
        self.cov = covariance #D @ C @ D
        
    def compute_mean_var(self):
        self.shift_returns_mean = self.shift_returns.ewm(span=self.shift).mean()
        self.shift_returns_var = self.shift_returns.ewm(span=self.shift).var()
        return self.shift_returns_mean, self.shift_returns_var
    
    def get_data(self):
        return self.prices
    
    def get_shift(self):
        return self.shift
    
    def get_mean_var_cov(self):

        self.compute_mean_var()
        self.compute_cov()
        return self.shift_returns_mean, self.shift_returns_var, self.cov

    '''-----------------------------------'''

    def download_data(self):
        '''
        download data with index_list from yahoo
        '''
        df = YahooDownloader(start_date = self.start_date,
                     end_date = self.end_date,
                     ticker_list = self.index_list).fetch_data()

        list_date = list(pd.date_range(df['date'].min(), df['date'].max()).astype(str))
        combination = list(itertools.product(list_date, self.index_list))

        processed_full = pd.DataFrame(combination, columns=['date', 'tic']).merge(df, on=['date', 'tic'], how='left')
        processed_full = processed_full[processed_full['date'].isin(df['date'])]
        processed_full = processed_full.sort_values(['date', 'tic'])

 

        self.prices = processed_full

    def generate_df(self):
        '''
        modify data
        '''
        self.download_data()
        processed = self.prices.reset_index().set_index(["tic", "date"]).sort_index()
        data_df = pd.DataFrame()
        for ticker in self.index_list:
            series = processed.xs(ticker).close
            data_df[ticker] = series
        data_df = data_df.reset_index()
        data_df.set_index("date", inplace=True)
        data_df.fillna(method="ffill", inplace=True)
        
        self.prices = data_df
        self.calc_return()

    def calc_return(self):
        result_df = self.prices.pct_change()
        result_df.columns = [str(col) + '_returns' for col in result_df.columns]        
        self.returns = result_df.fillna(0)[2:]
        self.prices = self.prices[2:]

    def split_series(self, prices, returns):
        prices = np.array(prices)
        returns = np.array(returns)
        
        X, y = list(), list()
        for window_start in range(prices.shape[0]):
            window_end = window_start + self.n_past #0+10
            future = window_end + 1 #10+1
            if future > prices.shape[0]:
                break
            # slicing the X window and y
            Xwindow, yy = prices[window_start:window_end, :], returns[window_end, :]
            X.append(Xwindow)
            y.append(yy)
        return np.array(X), np.array(y)

    def get_train_test_data(self, test_size = 0.3, normalize=True):

        X_price_train, X_price_valid, y_returns_train, y_returns_valid = train_test_split(self.prices, self.returns, test_size=test_size, shuffle=False)
        X_price_valid,  X_price_test, y_returns_valid, y_returns_test = train_test_split(X_price_valid,y_returns_valid, test_size = 0.5, shuffle=False)
        
        if normalize:
            scaler = StandardScaler()
            X_price_train = scaler.fit_transform(X_price_train) 
            X_price_test = scaler.transform(X_price_test)
                
        self.X_train, self.y_train = self.split_series(X_price_train, y_returns_train)
        self.X_valid, self.y_valid = self.split_series(X_price_valid, y_returns_valid)
        self.X_test, self.y_test = self.split_series(X_price_test, y_returns_test)

        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test #

    '''to-do: get valid data'''
    def get_train_valid(self):
        self.X_train_valid, self.y_train_valid = np.vstack((self.X_train, self.X_valid)), np.vstack((self.y_train, self.y_valid))
        return self.X_train_valid, self.y_train_valid


    """optional: convert when return_data is True"""
    # def convert_return_data(self):
    #     self.X_train, self.X_valid, self.X_test = self.y_train[:-1].copy(), self.y_valid[:-1].copy(), self.y_test[:-1].copy()
    #     self.y_train, self.y_valid, self.y_test = self.y_train[1:], self.y_valid[1:], self.y_test[1:]
    #     self.X_train, self.y_train = self.split_series(self.X_train,self.y_train)
    #     self.X_valid, self.y_valid = self.split_series(self.X_valid,self.y_valid)
    #     self.X_test, self.y_test = self.split_series(self.X_test,self.y_test)

    '''Thea'''
    def plot_data(self, ax):
        prices = np.array(self.prices)
        for i in range(self.prices.shape[1]):
            ax.plot(prices[:,i],label=self.index_list[i])
        ax.legend(loc='upper left', fontsize=8)
        
    """to-do: basic info of data"""
    # def information(self):
    #     return {'seed': self.seed, 'data_len': self.data_len, 'size': self.size, 'n_past': self.n_past, 'eig': str(self.eig_low) + ':' + str(self.eig_high), 'sd': str(self.sd_low) + ':' + str(self.sd_high)}
