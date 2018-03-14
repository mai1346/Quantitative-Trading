#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:39:58 2018

@author: mai1346
"""

import pandas as pd
import DBhelper as DBh
import statsmodels.tsa.stattools as ts
import time

#import close price as dataframe
engine=DBh.connect('mai1346','87566766','127.0.0.1:3306','cnstock')
close=pd.read_sql("select * from cnstock.close  where date between '2010-01-01' and '2015-12-31'",engine)
#%%

#Since the LinAlgError: SVD did not converge, we fill the NaN with back and forward values
close.fillna(method='ffill', inplace=True)
close.fillna(method='bfill', inplace=True)
close.dropna(axis=1,how='all', inplace=True)
close=close[close.columns[1:]]
#get the name of every code in hs300
symbols=pd.read_sql('select code,name from cnstock.hs300',engine)
codes=close.columns

#print ts.adfuller(close['600909'],1)
#############################################
#method 1
t0 = time.time()
adfuller=close.apply(ts.adfuller,maxlag=1)
norootcode=adfuller.loc[adfuller.apply(lambda x: x[0]<x[4]['5%'])].index
t1 = time.time()
total = t1-t0
print 'method 1:',total
########################################
#method 2
t2 = time.time()
norootcode=[]
for code in codes:
    result=ts.adfuller(close[code],1)
    # Compare the t-statistic with 5% critical value
    if result[0]<result[4]['5%']:
        norootcode.append(code)
t3= time.time()

print 'method 2:',t3-t2
norootstock=symbols[symbols['code'].isin(norootcode)]
print 'Result of Unit root test with Adfuller:\n ',  norootstock

#%%
#####################################
#Calculate the Hurst exponent of every stock
from numpy import std, subtract, polyfit, sqrt, log
def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""

    # create the range of lag values
#    i = len(ts) // 2
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst Exponent from the polyfit output
    return poly[0] * 2.0

hursts=close.apply(hurst)
print hursts
meanrevert_code=hursts.loc[lambda x: x<0.5].index
meanrevert_stock=symbols[symbols['code'].isin(meanrevert_code)]
print 'Result of Mean Reverting Stocks by Hurst Exponent: \n',meanrevert_stock

#%%
############################
#Cointegration test between pairs of hs300 stocks
#print close.head()
closereturn=close.pct_change()
closereturn.fillna(method='bfill',inplace =True)

#print closereturn.head()
#print ts.coint(close['600000'],close['600008'],maxlag=1)[2][1]
print len(codes)
codeL,codeR=[],[]
for i in range(len(codes)):
    for j in range(i+1,len(codes)):
        result=ts.coint(close[codes[i]],close[codes[j]],maxlag=1)
        if result[0]<result[2][1]:
            codeL.append(codes[i])
            codeR.append(codes[j])
print codeL
#%%
print len(codeL)

print len(codeR)
#%%
#Display the result pairs
codeLdf=pd.DataFrame(codeL,columns=['code'])
codeRdf=pd.DataFrame(codeR,columns=['code'])
Rpart=pd.merge(codeRdf,symbols,how='left') # Pay attention to how merge method works!
Lpart=pd.merge(codeLdf,symbols)

print Lpart.join(Rpart,lsuffix='One', rsuffix='Two')


            

  


