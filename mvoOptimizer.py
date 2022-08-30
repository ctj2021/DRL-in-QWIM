import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
from numpy import linalg as la
from cvxpy import *

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import black_litterman, risk_models

from data import Data
class compare_strategy:
  def __init__(self, data, mvo_start_date, interest_rate, min_return):
      self.start_date = mvo_start_date
      self.interest_rate = interest_rate
      self.min_return = min_return
      self.price = data.get_data().dropna()
      self.shift = data.get_shift() 
      self.shift_returns = data.get_shift_returns()
      self.shift_returns_mean, self.shift_returns_var, self.covariance = data.get_mean_var_cov()
      self.StockList = self.price.columns.values.tolist() + ["InterestRate"]

      self.index = self.shift_returns.index
      self.start_index = self.index.get_loc(self.start_date)
      self.end_index= self.index.get_loc(self.shift_returns.index[-1])
  
  def set_start_date(self, start_date):
      self.start_date = start_date
      self.start_index = self.index.get_loc(self.start_date)


  def MarkowitzOpt(self, mean, variance, covariance, interest_rate, min_return):
      n = mean.size + 1                   # Number of assets (number of stocks + interest rate)
      
      mu = mean.values                    # Mean returns of n assets
      temp = np.full(n, interest_rate)
      temp[:-1] = mu
      mu = temp
          
      counter = 0
      Sigma = np.zeros((n,n))                 # Covariance of n assets
      for i in np.arange(n-1):
          for j in np.arange(i, n-1):
              if i==j:
                  Sigma[i,j] = variance[i]
              else:
                  Sigma[i,j] = covariance[counter]
                  Sigma[j,i] = Sigma[i,j]
                  counter+=1
      Sigma = self.nearestPD(Sigma)                # Converting covariance to the nearest positive-definite matrix
      
      # Ensuring feasability of inequality contraint
      if mu.max() < min_return:
          min_return = interest_rate
      
      w = Variable(n)                         # Portfolio allocation vector
      ret = mu.T*	w
      risk = quad_form(w, Sigma)
      min_ret = Parameter(nonneg=True)
      min_ret.value = min_return
      prob = Problem(Minimize(risk),          # Restricting to long-only portfolio
                    [ret >= min_ret,
                    sum(w) == 1,
                    w >= 0])
      prob.solve()
      return w.value

  def nearestPD(self, A):
      """Find the nearest positive-definite matrix to input
      A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
      credits [2].
      [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
      [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
      matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
      """

      B = (A + A.T) / 2
      _, s, V = la.svd(B)

      H = np.dot(V.T, np.dot(np.diag(s), V))

      A2 = (B + H) / 2

      A3 = (A2 + A2.T) / 2

      if self.isPD(A3):
          return A3

      spacing = np.spacing(la.norm(A))
      # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
      # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
      # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
      # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
      # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
      # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
      # `spacing` will, for Gaussian random matrixes of small dimension, be on
      # othe order of 1e-16. In practice, both ways converge, as the unit test
      # below suggests.
      I = np.eye(A.shape[0])
      k = 1
      while not self.isPD(A3):
          mineig = np.min(np.real(la.eigvals(A3)))
          A3 += I * (-mineig * k**2 + spacing)
          k += 1

      return A3

  def isPD(self, B):
      """Returns true when input is positive-definite, via Cholesky"""
      try:
          _ = la.cholesky(B)
          return True
      except la.LinAlgError:
          return False

  def blacklitter(self, S, start_index):
      viewdict = {"^GSPC": 0.01, '^DJI':0.02} #,
      #S = risk_models.sample_cov(prices_df.iloc[start_index:start_index+20])
      bl = black_litterman.BlackLittermanModel(S, pi="equal",absolute_views = viewdict, omega="default")
      weight = bl.optimize(0.7)
      weights = []
      for key,value in weight.items():
          weights.append(value)
      weights.append(0)
      
      return weights
  
  def Markowitz(self):      
      date_index_iter = self.start_index
      distribution = pd.DataFrame(index=self.StockList)
      returns = pd.Series(index=self.index)
      
      # Start Value
      total_value = 1.0
      returns[self.index[date_index_iter]] = total_value
      portfolio_alloc = np.array([1/(self.price.shape[1]+1)]*(self.price.shape[1]+1))
      
      while date_index_iter + self.shift < self.end_index:
          # Calculating portfolio return
          date = self.index[date_index_iter]
          portfolio_alloc = self.MarkowitzOpt(self.shift_returns_mean.loc[date], self.shift_returns_var.loc[date], self.covariance.loc[date], self.interest_rate, self.min_return)
          # print(portfolio_alloc)
          distribution[date] = portfolio_alloc

          total_value_temp = total_value
          date1 = self.index[date_index_iter]
          for i in range(self.shift):
            
            date2 = self.index[date_index_iter+1]
            temp1 = self.price.loc[date2]/self.price.loc[date1]
            temp1.loc[self.StockList[-1]] = self.interest_rate+1
            temp2 = pd.Series(np.array(portfolio_alloc.ravel()).reshape(len(portfolio_alloc)),index=self.StockList)
            date_index_iter += 1
            total_value_temp = np.sum(total_value*temp2*temp1)
            returns[self.index[date_index_iter]] = total_value_temp
          total_value=total_value_temp


      date1 = self.index[date_index_iter]
      while date_index_iter < self.end_index:
        date2 = self.index[date_index_iter+1]
        temp1 = self.price.loc[date2]/self.price.loc[date1]
        temp1.loc[self.StockList[-1]] = self.interest_rate+1
        temp2 = pd.Series(np.array(portfolio_alloc.ravel()).reshape(len(portfolio_alloc)),index=self.StockList)
        date_index_iter += 1
        total_value_temp = np.sum(total_value*temp2*temp1)
        returns[self.index[date_index_iter]] = total_value_temp
      returns = returns[np.isfinite(returns)]
      return returns

  def bl_model(self):
      date_index_iter = self.start_index
      distribution = pd.DataFrame(index=self.StockList)
      returns_bl = pd.Series(index=self.index)

      # Start Value
      total_value = 1.0
      returns_bl[self.index[date_index_iter]] = total_value
      portfolio_alloc = np.array([1/(self.price.shape[1]+1)]*(self.price.shape[1]+1))
      # print(portfolio_alloc)
      
      while date_index_iter + self.shift < self.end_index: #+ self.shift
          # date = self.index[date_index_iter]
          total_value_temp = total_value
          date1 = self.index[date_index_iter]
          for i in range(self.shift):
            
            # date1 = self.index[date_index_iter]
            date2 = self.index[date_index_iter+1]
            temp1 = self.price.loc[date2]/self.price.loc[date1]
            temp1.loc[self.StockList[-1]] = self.interest_rate+1
            temp2 = pd.Series(np.array(portfolio_alloc.ravel()).reshape(len(portfolio_alloc)),index=self.StockList)
            date_index_iter += 1
            total_value_temp = np.sum(total_value*temp2*temp1)
            returns_bl[self.index[date_index_iter]] = total_value_temp
          total_value = total_value_temp
            # total_value = np.sum(total_value*temp2*temp1)
            # Increment Date
            
            # returns_bl[self.index[date_index_iter]] = total_value


          # date = self.index[date_index_iter]

          S = risk_models.sample_cov(self.price.iloc[date_index_iter-self.shift:date_index_iter])

          """try:
            #portfolio_alloc = np.array(blacklitter(nearestPD(S), date_index_iter))
            try:
              portfolio_alloc = np.array(blacklitter(nearestPD(S), date_index_iter))
            except:
              portfolio_alloc = np.array(blacklitter(S, date_index_iter))
          except:
            portfolio_alloc = portfolio_alloc"""

          try:
            portfolio_alloc = np.array(self.blacklitter(self.nearestPD(S), date_index_iter-self.shif))
          except:
            try:
              portfolio_alloc = np.array(self.blacklitter(S, date_index_iter-self.shif))
            except:
              portfolio_alloc = portfolio_alloc
            
      date1 = self.index[date_index_iter]
      while date_index_iter < self.end_index:
        date2 = self.index[date_index_iter+1]
        temp1 = self.price.loc[date2]/self.price.loc[date1]
        temp1.loc[self.StockList[-1]] = self.interest_rate+1
        temp2 = pd.Series(np.array(portfolio_alloc.ravel()).reshape(len(portfolio_alloc)),index=self.StockList)
        date_index_iter += 1
        total_value_temp = np.sum(total_value*temp2*temp1)
        returns_bl[self.index[date_index_iter]] = total_value_temp


      returns_bl = returns_bl[np.isfinite(returns_bl)]

      return returns_bl