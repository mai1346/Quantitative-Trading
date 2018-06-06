# -*- coding: utf-8 -*-
"""
Created on Sat May 12 23:59:29 2018

@author: mai1346
"""
#%%
import tushare as ts
import pandas as pd
import StockDataScrape as inv
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as tsa
from pykalman import KalmanFilter
import numpy as np
#%% 获取symbol
symbolA = inv.getpairID('美的')[1]
symbolB = inv.getpairID('格力')[1]
#%%获取数据
stockA = ts.get_k_data(symbolA,start='2015-01-01',end='2015-12-31')
stockB = ts.get_k_data(symbolB,start='2015-01-01',end='2015-12-31')
bench=ts.get_k_data('399001',start='2015-01-01',end='2015-12-31')
stockA['date']=pd.to_datetime(stockA['date'])
stockA.set_index('date', inplace = True)
stockA.drop('code',axis= 1, inplace = True)
stockB['date']=pd.to_datetime(stockB['date'])
stockB.set_index('date', inplace = True)
stockB.drop('code',axis= 1, inplace = True)
#%% 可视化数据及协整检验
plt.figure(figsize=(8,4.5))
plt.plot(stockA.close,color='r')
plt.plot(stockB.close,color='b')
df=stockA.join(stockB.close,how='outer',rsuffix='b')
df=df[['close','closeb']]
df.fillna(method='ffill',inplace= True)
print (tsa.coint(df['close'],df['closeb'],maxlag=1))
#%%Kalman filter 生成动态的hedge ratio
obs_mat = np.vstack([df.closeb, np.ones(df.closeb.shape)]).T[:, np.newaxis]
delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)
kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=1.0,
                  transition_covariance=trans_cov)
state_means, state_covs = kf.filter(df.close)

dfparam=pd.DataFrame(state_means,index=df.index,columns=['slope','intercept'])
dfparam.plot(subplots = True,figsize =(8,4.5))
#%% 根据hedge ratio 计算spread和z-score
df['closehat']=df.closeb*dfparam.slope+dfparam.intercept
df['spread']=df.close-df.closehat
df['zscore']=(df.spread-df.spread.mean())/df.spread.std()
#对zscore 进行可视化
df.zscore.plot(grid = True)
#%%选取threshold，生成signal 和 pair position
longstatus = False
shortstatus = False
upbound = 3.5
lowbound = 0.2
signal=[]

for z in df.zscore.values:
    if z > upbound and not longstatus: #enter long
        signal.append(1)
        longstatus = True
    elif z < -upbound and not shortstatus: #enter short
        signal.append(-1)
        shortstatus = True
    elif abs(z) < lowbound and longstatus: #exit long 
        signal.append(-1)
        longstatus= False
    elif abs(z) > lowbound and shortstatus: #exit short
        signal.append(1)
        shortstatus = False
    else:
        signal.append(0)

df['signal'] = signal
#生成position
df['positionA'] = -df.signal.cumsum()
df['positionB'] = df.signal.cumsum()
#%%计算portfolio 及benchmarkreturn和performance
df['returnA']=df.close.pct_change()
df['returnB']=(df.closeb*dfparam.slope).pct_change()
df['Strategy Return']=df['returnA']*df['positionA']+df['returnB']*df['positionB']

df['Strategy Equity']=(df['Strategy Return']+1).cumprod()
df['Strategy Drawdown']=df['Strategy Equity']/df['Strategy Equity'].expanding().max()-1


df['Bench Return'] = bench['close'].pct_change().values
df['Bench Equity']=(df['Bench Return']+1).cumprod()
df['Bench Drawdown']=df['Bench Equity']/df['Bench Equity'].expanding().max()-1


df.dropna(inplace = True)
ststd=df['Strategy Return'].std()
stavg=df['Strategy Return'].mean()
stsharpe=stavg/ststd*np.sqrt(252)
maxdrawdown= min(df['Strategy Drawdown'])
benstd=df['Bench Return'].std()
benavg=df['Bench Return'].mean()
bensharpe=benavg/benstd*np.sqrt(252)
benchdrawdown=min(df['Bench Drawdown'])

#%%输出结果和可视化
print('Strategy Return: %3.4f%%' % (df['Strategy Equity'][-1]*100-100),
      '\nStrategy Average Daily Return:%3.4f%%' % (stavg*100), 
      '\nStrategy Std: %3.4f%%' % (ststd*100), 
      '\nStrategy Sharpe Ratio: %3.4f' % stsharpe, 
      '\nStrategy Max Drawdown: %3.4f%%' % (maxdrawdown*100))
print('Benchmark Return: %3.4f%%' % (df['Bench Equity'][-1]*100-100),
      '\nBenchmark Average Daily Return:%3.4f%%' % (benavg*100), 
      '\nBenchmark Std: %3.4f%%' % (benstd*100), 
      '\nBenchmark Sharpe Ratio: %3.4f' % bensharpe,
      '\nBenchmark Max Drawdown: %3.4f%%' % (benchdrawdown*100))
df[['Bench Equity','Strategy Equity']].plot(figsize=(16,9),grid= True)