# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:43:55 2018

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
symbolA = inv.getpairID('茅台',source = 'cn')[1]
symbolB = inv.getpairID('五粮液',source = 'cn')[1]
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
#%%选取threshold，并根据不同threshold组合生成不同的signal以及pair positions
highs=np.linspace(2,4,21)
lows = np.linspace(0.1,0.5,5)

def genSignal(highs,lows,df):
    for high in highs:
        for low in lows:  
            longstatus = False
            shortstatus = False
            signal=[]
            for z in df.zscore.values:
                if z > high and not longstatus: #enter long
                    signal.append(1)
                    longstatus = True
                elif z < -high and not shortstatus: #enter short
                    signal.append(-1)
                    shortstatus = True
                elif abs(z) < low and longstatus: #exit long 
                    signal.append(-1)
                    longstatus= False
                elif abs(z) > low and shortstatus: #exit short
                    signal.append(1)
                    shortstatus = False
                else:
                    signal.append(0)
            
            df['signal(%1.1f,%1.1f)' % (high,low)] = signal
            df['positionA(%1.1f,%1.1f)' % (high,low)] = -df['signal(%1.1f,%1.1f)' % (high,low)].cumsum()
            df['positionB(%1.1f,%1.1f)' % (high,low)] = df['signal(%1.1f,%1.1f)' % (high,low)].cumsum()
genSignal(highs,lows,df)
#%%计算threshold参数的portfolio return和最大回撤
df['returnA']=df.close.pct_change()
df['returnB']=(df.closeb*dfparam.slope).pct_change()
for high in highs:
    for low in lows:  
        df['Strategy Return(%1.1f,%1.1f)' % (high,low)]=df['returnA']*df['positionA(%1.1f,%1.1f)' % (high,low)]\
        +df['returnB']*df['positionB(%1.1f,%1.1f)' % (high,low)]
        df['Strategy Equity(%1.1f,%1.1f)' % (high,low)]=(df['Strategy Return(%1.1f,%1.1f)' % (high,low)]+1).cumprod()
        df['Strategy Drawdown(%1.1f,%1.1f)' % (high,low)]=df['Strategy Equity(%1.1f,%1.1f)' % (high,low)]/df['Strategy Equity(%1.1f,%1.1f)' % (high,low)].expanding().max()-1

#%%计算benchmark的Performance
df['Bench Return'] = bench.close.pct_change().values
df['Bench Equity']=(df['Bench Return']+1).cumprod()
df['Bench Drawdown']=df['Bench Equity']/df['Bench Equity'].cummax()-1

#%%画出所有threshold组合的equity curve并筛选出return最高，回撤最小的组合。
df1=df.dropna()
pairs=[]
for high in highs:
    for low in lows: 
        df['Strategy Equity(%1.1f,%1.1f)' % (high,low)].plot(figsize=(16,9),grid= True)
        pairs.append(((high,low),df1['Strategy Equity(%1.1f,%1.1f)' % (high,low)][-1],min(df1['Strategy Drawdown(%1.1f,%1.1f)' % (high,low)])))

pairs.sort(key=lambda x:(-x[1], x[2]))
df1['Bench Equity'].plot()
#%%计算最佳threshold组合的策略Performance
besthigh, bestlow = pairs[0][0]

ststd=df1['Strategy Return(%1.1f,%1.1f)'% (besthigh,bestlow)].std()
stavg=df1['Strategy Return(%1.1f,%1.1f)'% (besthigh,bestlow)].mean()
stsharpe=stavg/ststd*np.sqrt(252)
maxdrawdown=pairs[0][2]
benstd=df1['Bench Return'].std()
benavg=df1['Bench Return'].mean()
bensharpe=benavg/benstd*np.sqrt(252)
benchdrawdown=min(df1['Bench Drawdown'])

#%%输出最佳策略结果和可视化
print('Best Strategy with Upper and Lower bound: %s' % str(pairs[0][0]),
      '\nStrategy Return: %3.4f%%' % (pairs[0][1]*100-100),
      '\nStrategy Average Daily Return:%3.4f%%' % (stavg*100), 
      '\nStrategy Std: %3.4f%%' % (ststd*100), 
      '\nStrategy Sharpe Ratio: %3.4f' % stsharpe, 
      '\nStrategy Max Drawdown: %3.4f%%' % (maxdrawdown*100))
print('Benchmark Return: %3.4f%%' % (df['Bench Equity'][-1]*100-100),
      '\nBenchmark Average Daily Return:%3.4f%%' % (benavg*100), 
      '\nBenchmark Std: %3.4f%%' % (benstd*100), 
      '\nBenchmark Sharpe Ratio: %3.4f' % bensharpe,
      '\nBenchmark Max Drawdown: %3.4f%%' % (benchdrawdown*100))
df1[['Bench Equity','Strategy Equity(%1.1f,%1.1f)'% (besthigh,bestlow)]].plot(figsize=(16,9),grid= True)
