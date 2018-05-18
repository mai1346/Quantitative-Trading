# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 09:08:50 2018

@author: Haoyuan
"""

import datetime
import sqlalchemy as sqla
import tushare as ts
import pandas as pd


def connect(**login):
    """connect to a database use information provided
    args:
        username: MySQL username
        password: password
        host: host address
        db: name of the database
    return:
        a sqlalchemy engine connection
    """
    engine=sqla.create_engine('mysql://%s:%s@%s/%s?charset=utf8' % 
                             (login['username'],login['password'],login['host'],login['db']))
    return engine
#
def obtain_and_insert_hs300(con):
    """Get the composite of hs300 index and insert it to database
       args:
           con: a sqlalchemy engine connection
       return: the code of every stock and 
           a database table containing hs300 stocks and ticker
    """
    # Get hs300 from tushare
    hs300=ts.get_hs300s()
    hs300['date']=pd.to_datetime(hs300['date'])
    hs300.set_index('date',inplace=True)
    # write it to MySQl database named 'cnstock'

    hs300.to_sql('hs300',con, if_exists='replace') #dtype={'date':sqla.types.VARCHAR(12)}


def obtain_daily_data(con,table,stocks,start='2009-01-01',\
                      end=datetime.date.today().isoformat()):
    """Get daily stock data of hs300 stocks and storing it in MySQL table named 'price',
       if exists, old table will be replaced.
       args: 
           con: a sqlalchemy engine connection
           table: a string, target table
           stock: a list, of stocks' code to fetch
           start: formatted date, beginning date of historical price
           end: ending date 
       output: 
            database tables named after the stock code, each containing stock daily prices 
            during the inputed time span
    """
    try:
        con.execute('DROP TABLE cnstock.%s' %(table))
    except:
        pass
    num=1 #for progress display use 
    for s in stocks:
        daily=ts.get_k_data(str(s),start=start,end=end)
        daily.to_sql(str(table),con,if_exists='append')
        # show adding progress
        print ('Stock:%s added to database, progress: %d/%d %4.2f%% complete.' \
        % (s,num,len(stocks),float(num)*100/len(stocks)))
        num+=1
#        if num==50:
#            break
def clean_data(con,openT='open',closeT='close'):
    """
    This function did two things below:
    1. Transform the raw price data to a dataframe with a shape of i*j,
    where i represents time point, j represent the stock numbers.
    2. Write the new dataframe back to database.
    args:
        con: a sqlalchemy engine connection
        openT: name of open price table
        closeT: name of close price table
    """
    DB=pd.read_sql("select date,close,open,code from cnstock.rawdata",con)
    code=pd.read_sql('SELECT code from cnstock.hs300',con)['code']
    DB['date']=pd.to_datetime(DB['date'])
    DB.set_index('date',inplace=True)
    seriesclose,seriesopen=[],[]
    for s in code:
        seriesclose.append(DB[DB['code']==s]['close'])
        seriesopen.append(DB[DB['code']==s]['open'])
    newclose=pd.concat(seriesclose,axis=1,keys=code)
    newopen=pd.concat(seriesopen,axis=1,keys=code)
    newopen.to_sql(str(openT),con,if_exists='replace')
    newclose.to_sql(str(closeT),con,if_exists='replace')


#%%
if __name__ =='__main__':
    login ={'username':'stockuser','password':'87566766','host':'localhost','db':'cnstock'}
    engine =connect(**login)
    obtain_and_insert_hs300(engine)
   # codes=pd.read_sql('select code from cnstock.hs300',engine)['code']
   # obtain_daily_data(engine,'rawdata',codes)
   # clean_data(engine)
    
    
