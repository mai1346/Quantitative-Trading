# -*- coding: utf-8 -*-
"""
Created on Sun May  6 09:25:47 2018

@author: mai1346
"""

import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
import seaborn
import numpy as np
#%% Load data
start='2015-01-01'
end='2018-01-01'
df=ts.get_k_data('600050', start = start)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace = True)
df.drop('code',axis=1, inplace = True)
print (df.head())

#%% Generate Signal
#sma=df.open.rolling(5).mean()
#lma=df.open.rolling(10).mean()
sma=df.open.ewm(span=5).mean()
lma=df.open.ewm(span=10).mean()

df['signal']=0
buy= (sma > lma) & (sma.shift() < lma.shift())
sell= (sma < lma) & (sma.shift() > lma.shift())
first=buy.loc[buy==True]

df['signal']+=buy
#df['signal']-=sell
#df['signal'][first.index[0]:]-=sell
df.loc[first.index[0]:,'signal']-=sell
#%%Calculate Equity Value and Performance
df['Bench Return']=df['open'].pct_change()
df['position']=df.signal.cumsum()
df['Strategy Return']=df['Bench Return']*df['position']
df['Bench Equity']=(df['Bench Return']+1).cumprod()
df['Strategy Equity']=(df['Strategy Return']+1).cumprod()
df['Strategy Drawdown']=df['Strategy Equity']/df['Strategy Equity'].expanding().max()-1
df['Bench Drawdown']=df['Bench Equity']/df['Bench Equity'].expanding().max()-1
df.dropna(inplace= True)
benstd=df['Bench Return'].std()
benavg=df['Bench Return'].mean()
bensharpe=benavg/benstd*np.sqrt(252)
benchdrawdown=min(df['Bench Drawdown'])

ststd=df['Strategy Return'].std()
stavg=df['Strategy Return'].mean()
stsharpe=stavg/ststd*np.sqrt(252)
maxdrawdown= min(df['Strategy Drawdown'])

#%% Summary and Plot
print('Benchmark Return: %3.4f%%' % (df['Bench Equity'][-1]*100-100),
      '\nBenchmark Average Daily Return:%3.4f%%' % (benavg*100), 
      '\nBenchmark Std: %3.4f%%' % (benstd*100), 
      '\nBenchmark Sharpe Ratio: %3.4f' % bensharpe,
      '\nBenchmark Max Drawdown: %3.4f%%' % (benchdrawdown*100))
print('Strategy Return: %3.4f%%' % (df['Strategy Equity'][-1]*100-100),
      '\nStrategy Average Daily Return:%3.4f%%' % (stavg*100), 
      '\nStrategy Std: %3.4f%%' % (ststd*100), 
      '\nStrategy Sharpe Ratio: %3.4f' % stsharpe, 
      '\nStrategy Max Drawdown: %3.4f%%' % (maxdrawdown*100))


df[['Bench Equity','Strategy Equity']].plot(figsize=(16,9),grid= True)