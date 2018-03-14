# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 11:59:34 2018

@author: Haoyuan
"""

import datetime
import pandas as pd
import sqlalchemy as sqla
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold
import json
def connect(username,password,host,db):
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
                             (username,password,host,db))
    return engine

def obtain_and_insert_hs300(con):
    """Get the composite of hs300 index and insert it to database
       args:
           con: a sqlalchemy engine connection
       return: the code of every stock and 
           a database table containing hs300 stocks and ticker
    """
    # Get hs300 from tushare
    hs300=ts.get_hs300s()
    # write it to MySQl database named 'cnstock'
    hs300.to_sql('hs300',con, if_exists='replace',dtype={'date':sqla.types.VARCHAR(12)}) #dtype={'date':sqla.types.VARCHAR(12)}


def obtain_daily_data(con,table,stocks,start='2009-01-01',\
                      end=datetime.date.today().isoformat()):
    """Get daily stock data of hs300 stocks and storing it in MySQL table named 'price'
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
    num=1 #for progress display use 
    for s in stocks:
        daily=ts.get_k_data(str(s),start=start,end=end)
        daily.to_sql(str(table),con,if_exists='append',dtype={'date':sqla.types.VARCHAR(12)})
        # show adding progress
        print 'Stock:%s added to database, progress: %d/%d %4.2f%% complete.' \
        % (s,num,len(stocks),float(num)*100/len(stocks))
        num+=1
#        if num==50:
#            break
        

def read_daily_data(con,table,codes,start='2015-01-01',\
                    end=datetime.date.today().isoformat()):
    """ Read from MySQL database the required quotes of hs300 stocks
        args:
            con: a sqlalchemy engine connection
            stock: a list of stocks' code to fetch
            start: beginning date of historical price
            end: ending date 
        return:
            daily variation of stock price;
            valid stock code (for the purpose of same sample size)
    """
    quotes=[]
    price=pd.read_sql("select date,close,open,code from cnstock.%s where date \
                      between '%s' and '%s'" %(table,start,end),con)
    usedstock=[]
    samplelen=[]
    #find the mode of sample size
    for s in codes:
        samplelen.append(len(price[price['code']==s]))
#    print samplelen
    modelen=max(samplelen,key=samplelen.count)
    print 'Valid Sample Size= %s' % (modelen)
    for s in codes: 
        if len(price[price['code']==s])==modelen:
            print s+'added'
            quotes.append(price[price['code']==s])
            usedstock.append(str(s))

    close_prices = np.vstack([q['close'] for q in quotes])
    print close_prices, close_prices.shape
    open_prices = np.vstack([q['open'] for q in quotes])
    variation=close_prices-open_prices
    return variation,usedstock

engine=connect('root','wo10who','127.0.0.1:3306','cnstock')

#print json.dumps(symbols, encoding="UTF-8", ensure_ascii=False)
########################################################
symbols=pd.read_sql('SELECT code, name from cnstock.hs300',engine)
code=symbols['code']
#symbolsdict=dict(zip(symbols['code'],symbols['name']))
variation,validstock=read_daily_data(engine,'rawdata',code,start='2017-07-01',end='2017-12-31')
print 'Valid Stock Number: %d' % (len(validstock))
validsymbols=symbols[symbols['code'].isin(validstock)]
codes, names=validsymbols.as_matrix().T
print variation.dtype

 #############################################################################
# Learn a graphical structure from the correlations
edge_model = covariance.GraphLassoCV()

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery

X = variation.copy().T
print X
print X.dtype
X /= X.std(axis=0)
edge_model.fit(X)
#%%
## #############################################################################
# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

for i in range(n_labels + 1):
    print'Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i]))
    
    
 #############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane
#
# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

# #############################################################################
# Visualization
plt.figure(1, facecolor='w', figsize=(30, 30))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=16,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6),fontproperties = 'SimHei')

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()    
##%%
#engine=connect('root','wo10who','localhost','cnstock')
#code=ts.get_hs300s()[['code']]
#stock=list(code.values.reshape(1,-1)[0])
#for s in stock:
#    try:
#        engine.execute('DROP TABLE cnstock.%s' %(s))
#    except:
#        continue