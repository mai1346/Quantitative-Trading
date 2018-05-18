#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:52:14 2018

@author: mai1346
"""

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import FeatureUtility as FUt
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
#%%
# Help Functions to process stock data into feature
def CalTech(dataframe,timespan):
    '''This function calculates technical indicators of a stock, including CCI,EVM,SMA,
       EWMA,BBAND,FORCEINDEX and ROC.
       
       Input:
           dataframe: One stock's trading data including at least its open, close, high, low and volume.
           timespan: Timespan used to calculate technical indicators.
       Output:
           A dataframe including these indicators that can be used as features.
    '''
    feature=dataframe.copy()
    for func in FUt.TechF:
        feature=func(feature,timespan)
    feature.dropna(axis=0,how='any',inplace= True)
#    feature.drop(['open','close','high','low','volume','code'], axis=1, inplace= True)
    return feature

def Scaling(dataframe,method='MinMax'):
    '''Scaling features using given method.

       Input:
           dataframe: feature dataframe
           method:
               method=MinMax, use Rescaling x^=(x-min(x))/(max(x)-min(x))
               method=standardization, use Standardization x^=(x-mean(x))/std(x)
       Output:
            Drop non-feature columns and Add scaling features to orginal dataframe
    '''
    def MinMax(series):
        return (series-series.min())/(series.max()-series.min())
    
    def standardization(series):
        return (series-series.mean())/series.std()
    
    scale=dataframe.drop(['open','close','high','low','volume','code'], axis=1)
    
    if method=='MinMax':
        result=scale.apply(MinMax)        
#        for column in scale.columns:
#            scale[column+'_Scaled']=(scale[column]-scale[column].min())/(scale[column].max()-scale[column].min())
    if method=='standardization':
        result=scale.apply(standardization)
#        for column in scale.columns:
#            scale[column+'_Scaled']=(scale[column]-scale[column].mean())/scale[column].std()
    return result
    
def LabelGen(dataframe):
    '''This function generate label or y used in the model. By now, it returns two kinds of Y.
       1. Whether the stock price will increase or decrease in the next day. 
          Increase: labeled as 1
          Decrease: labeled as -1
       2. The close price in the next day. Used in the regression tasks.
       
       Input:
           dataframe: One stock's trading data including at least its close price.
       Output:
           A dataframe contains classification labels and value in corresponding columns.
    '''
    price_change=pd.DataFrame(dataframe['close'].pct_change())

    price_change['Y_label']=price_change.apply(np.sign)

#    price_change['Y_label']=np.where(price_change['close']>0,1,-1)
    price_change['Y_label']=price_change['Y_label'].shift(-1)
    
#    price_change.dropna(axis=0,how='any',inplace= True)
    price_change['Y_value']=dataframe['close']
    price_change.drop('close',axis=1,inplace=True)
    return price_change
    

    
#%%
# Get Hs300 index data
start='2013-01-01'
end='2018-01-01'
hs300=ts.get_k_data('hs300',start=start,end=end)
hs300.set_index('date',inplace=True)

#%%
# Process data: Feature Space
Techfea=CalTech(hs300,20)
sp500=FUt.SP500(start,end)
rate=FUt.TenYearRate()
feature=Techfea.join(sp500).join(rate)
#feature=feature.drop(['open','close','high','low','volume','code'], axis=1)
#min_max_scaler = preprocessing.MinMaxScaler().fit(feature)
#scaledfeature = min_max_scaler.transform(feature)
scaledfeature=Scaling(feature)
# Process data: Output Space
Output=LabelGen(hs300)

data=scaledfeature.join(Output)
data.dropna(axis=0,how='any',inplace= True)
#%%
X=data[data.columns[:-2]]

Y_label=data['Y_label']
Y_value=data['Y_value']

#%%
# Plot the feature importances of the forest


forest = ExtraTreesClassifier(n_estimators=250,random_state=0)

forest.fit(X, Y_label)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print "Feature ranking:"

for f in range(X.shape[1]):
    print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])


plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
#%% Classification Task
# Cross Validation

k_fold = KFold(n_splits=3)

#%% RandomForest
import time
from sklearn.model_selection import GridSearchCV

t0=time.time()
# Loop method
#T_range = range(10, 400, 20)  
#  
#T_scores = []  
#  
#  
#for t in T_range:  
#    RF = ExtraTreesClassifier(n_estimators=t,random_state=0)  
#    scores = cross_val_score(RF, X, Y_label, cv=k_fold, scoring='accuracy',n_jobs=-1)  
#    T_scores.append(scores.mean())  

# GridSearch
tuned_parameters={'n_estimators':range(10,400,20)}
model = GridSearchCV(ExtraTreesClassifier(), tuned_parameters, cv=k_fold, n_jobs=-1)
model.fit(X,Y_label)

means=model.cv_results_['mean_test_score']
params=model.cv_results_['params']
print "Grid scores calculated on training set:"
for mean,param in zip(means,params):
    print "%0.3f for %r" % (mean, param)
t1=time.time()


print t1-t0
#可视化数据  
#plt.plot(T_range, T_scores)  
#plt.xlabel('Number of Trees for Random Forest')  
#plt.title('Cross-Validated Accuracy')  
#plt.show()  

#%% Logistic Regression
LR= LogisticRegression(C=1000)
Lscore=cross_val_score(LR,X,Y_label, cv=k_fold, scoring= 'accuracy')
print Lscore.mean()
plt.plot(Lscore)
#%% SVM
svc = svm.SVC(C=1, kernel='linear')
SVCscore=cross_val_score(svc,X,Y_label, cv=k_fold, scoring= 'accuracy')
print SVCscore.mean()
plt.plot(SVCscore)

#%% Regression Task


x_train, x_test, y_train, y_test = train_test_split(X, Y_value, test_size=0.33,shuffle= False)

models = [("RF", ExtraTreeRegressor(random_state=0)),
          ("LR", LinearRegression(n_jobs=-1))]

for m in models:
    m[1].fit(x_train, y_train)

    # Make an array of predictions on the test set
    pred = m[1].predict(x_test)

    # Output the hit-rate and the confusion matrix for each model
    print("%s:\n%0.6f" % (m[0], m[1].score(x_test, y_test)))

    result=pd.DataFrame(index=y_test.index)
    result['y_pred']=pred
    result['y_test']=y_test
    #Linscore=cross_val_score(LinRe,X,Y_value, cv=k_fold, scoring= 'r2')
    #print Linscore.mean()
    result.plot(figsize=(16,9),title='%s' %(m[0]))

#%% Ensemble Method Regression

ada=AdaBoostRegressor(ExtraTreeRegressor(),n_estimators=300, random_state=0)
ada.fit(x_train,y_train)
pred=ada.predict(x_test)
print ada.score(x_test,y_test)
result=pd.DataFrame(index=y_test.index)
result['y_pred']=pred
result['y_test']=y_test
result.plot(figsize=(16,9))

#%%

x_train, x_test, y_train, y_test = train_test_split(X, Y_label, test_size=0.33,shuffle= False)
#RFclf=ExtraTreesClassifier(n_estimators=250,random_state=0,n_jobs=-1)
RFclf=svm.SVC(C=1, kernel='rbf')
RFclf.fit(x_train,y_train)
print 'RF without further ensemble', RFclf.score(x_test,y_test)
#adaclf=AdaBoostClassifier(ExtraTreesClassifier(n_estimators=250,random_state=0,n_jobs=-1),n_estimators=300, algorithm='SAMME',random_state=0)
adaclf=AdaBoostClassifier(svm.SVC(C=1, kernel='rbf'),n_estimators=1000, algorithm='SAMME',random_state=0)
adaclf.fit(x_train,y_train)
print 'RF with further ensemble', adaclf.score(x_test,y_test)