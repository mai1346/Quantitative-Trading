#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:36:39 2018

@author: mai1346
"""
import pandas as pd
import tushare as ts
import pandas_datareader as web



def CCI(data, ndays): 
    TP = (data['high'] + data['low'] + data['close']) / 3 
    CCI = pd.Series((TP - TP.rolling(window=ndays,center=False).mean()) / (0.015 * TP.rolling(window=ndays,center=False).std()),name = 'CCI') 
    data = data.join(CCI) 
    return data
 
def EVM(data, ndays): 
    dm = ((data['high'] + data['low'])/2) - ((data['high'].shift(1) + data['low'].shift(1))/2)
    br = (data['volume'] / 100000000) / ((data['high'] - data['low']))
    EVM = dm / br 
    EVM_MA = pd.Series(EVM.rolling(window=ndays,center=False).mean(), name = 'EVM') 
    data = data.join(EVM_MA) 
    return data

# Simple Moving Average 
def SMA(data, ndays): 
    SMA = pd.Series(data['close'].rolling(window=ndays,center=False).mean(), name = 'SMA') 
    data = data.join(SMA) 
    return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
    EMA =pd.Series(data['close'].ewm(ignore_na=False,span=ndays,min_periods=ndays-1,adjust=True).mean(),name = 'EWMA_' + str(ndays))
    data = data.join(EMA) 
    return data

def BBANDS(data, ndays):

    MA = pd.Series(data['close'].rolling(window=ndays,center=False).mean()) 
    SD = pd.Series(data['close'].rolling(window=ndays,center=False).std())
    
    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name = 'Upper_BollingerBand') 
    data = data.join(B1) 
     
    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name = 'Lower_BollingerBand') 
    data = data.join(B2) 
     
    return data

def FI(data, ndays): 
    FI = pd.Series(data['close'].diff(ndays) * data['volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data

def ROC(data,n):
    N = data['close'].diff(n)
    D = data['close'].shift(n)
    ROC = pd.Series(N/D,name='Rate_Of_Change')
    data = data.join(ROC)
    return data 

def SP500(start,end):
    sp500=web.DataReader('SPX','morningstar',start,end)
    sp500.drop(['Open','High','Low','Volume'],axis=1,inplace=True)
    sp500.reset_index(level='Symbol',drop= True, inplace=True)
    sp500.index.rename('date',inplace=True)
    sp500.rename(columns={'Close':'Sp500'}, inplace=True)
    sp500['Sp500']=sp500['Sp500'].shift(1)

    return sp500

def TenYearRate():
    Rate=pd.read_csv('10yearrate.csv',encoding='utf-8')
    close=Rate[[u'日期',u'最新']]
    close.rename(columns={u'日期':'date',u'最新':'10yearRate'}, inplace= True)
    close.loc[:,'date']=pd.to_datetime(close.loc[:,'date'])
    close.set_index('date',inplace= True)
    close=close[::-1]
    return close

TechF=[CCI,EVM, SMA, EWMA, BBANDS,FI, ROC]

if __name__=='__main__':
#    sp=SP500(start='2009-01-01',end='2018-01-01')
#    print sp.head()
    rate=TenYearRate()