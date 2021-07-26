# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:52:19 2021

@author: pphung
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import os
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier
import random
import matplotlib.pyplot as plt
import seaborn as sns


path = os.getcwd()


# load data

adm = gpd.read_file(os.path.abspath(os.path.join("../..", "Model (ENSO)/M1/shape/zwe_admbnda_adm2_zimstat_ocha_20180911.shp")))
crop_anomaly = pd.read_csv(os.path.abspath(os.path.join("../..", "Model (ENSO)/M1/data/zwe_crop_yield_anomaly.csv")))
precip_obs = pd.read_csv(os.path.abspath(os.path.join("../../..", "3. Data - Hazard exposure, vulnerability/zwe_rain/zwe_monthcumulative_rainfall.csv")))
# # precip_fc = xr.open_dataset(os.path.abspath(os.path.join("..", "input/monthly_ensamble_mean_1993_2016.nc")))
# # precip_fc = xr.open_dataset(os.path.abspath(os.path.join("../../..", "3. Data - Hazard exposure, vulnerability/zwe_rainforecast/adaptor.mars.external-1625072457.3784559-2573-2-9db01f17-9950-4c9f-b016-b2e4c6245f97.grib")),
# #                                             engine='cfgrib')
# precip_fc1 = xr.open_dataset(os.path.abspath(os.path.join("../../..", "3. Data - Hazard exposure, vulnerability/zwe_rainforecast/1.nc")))
# precip_fc2 = xr.open_dataset(os.path.abspath(os.path.join("../../..", "3. Data - Hazard exposure, vulnerability/zwe_rainforecast/2.nc")))
# precip_fc1 = precip_fc1.to_dataframe()
# # precip_fc = xr.merge([precip_fc1, precip_fc2])



# preprocess data

df = pd.DataFrame(data={'year': np.arange(2000, 2021)})

# precip_obs['date'] = pd.to_datetime(precip_obs["year"].astype(str) + '-' + precip_obs["month"].astype(str), format='%Y-%m')
precip_obs = precip_obs.rename(columns={'year':'year_original'})
precip_obs['year'] = np.where(precip_obs['month']>=7, precip_obs['year_original']+1, precip_obs['year_original']+0)
precip_obs = pd.pivot_table(precip_obs, values = 'pcumul', index=['pcode','year'], columns = 'month').reset_index()
precip_obs = precip_obs[precip_obs['year'] != 1999]


crop_anomaly['date'] = pd.to_datetime(crop_anomaly['date'])
crop_anomaly['year'] = crop_anomaly['date'].dt.year
crop_anomaly = crop_anomaly.rename(columns={'ADM2_PCODE':'pcode'})

df = pd.merge(precip_obs, crop_anomaly, on=['year', 'pcode'], how='left')

df['drought'] = np.where(df['anomaly']=='Yes', 1, 0)
df = df.drop(columns=['Unnamed: 0', 'date', 'remainder', 'anomaly'])
# df = df[df['year'] <= 2019]

df_7month = df.drop(columns=[10,11,12,1,2,3])
df_6month = df.drop(columns=[11,12,1,2,3])
df_6month['cumul'] = df_6month[[9,10]].sum(axis=1)
df_5month = df.drop(columns=[12,1,2,3])
df_5month['cumul'] = df_5month[[9,10,11]].sum(axis=1)
df_4month = df.drop(columns=[1,2,3])
df_4month['cumul'] = df_4month[[9,10,11,12]].sum(axis=1)
df_3month = df.drop(columns=[2,3])
df_3month['cumul'] = df_3month[[9,10,11,12,1]].sum(axis=1)
df_2month = df.drop(columns=3)
df_2month['cumul'] = df_2month[[9,10,11,12,1,2]].sum(axis=1)
df_1month = df
df_1month['cumul'] = df_1month[[9,10,11,12,1,2,3]].sum(axis=1)


df_list = [df_7month, df_6month, df_5month, df_4month,
           df_3month, df_2month, df_1month]
leadtimes = np.arange(7,0,-1)


#%% 1 MONTH LEADTIME set up data for xgboost model


leadtime = leadtimes[6]
# df_predict_1 = pd.DataFrame()
i = 0

# X_train = pd.DataFrame()
# X_test = pd.DataFrame()
# y_train = pd.DataFrame()#pd.Series()
# y_test = pd.DataFrame()#pd.Series()
labels = np.unique(df['drought'])   


# split train-test set per district
# for district in df['pcode'].unique():

X = df_1month.drop(columns=['pcode', 'year', 'drought'])
X = StandardScaler().fit_transform(X) # normalize features
X = pd.DataFrame(data=X, columns=[1,2,3,9,10,11,12, 'cumul'])
y = df_1month[['drought']]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, 
                                                    stratify=y,
                                                    random_state=42)
# X_train = X_train.append(X_train_dist.drop(columns=['pcode']))
# X_test = X_test.append(X_test_dist)
# y_train = y_train.append(y_train_dist.drop(columns=['pcode']), ignore_index=True)
# y_test = y_test.append(y_test_dist, ignore_index=True)



# parametres

model = XGBClassifier(objective='binary:logistic',
                      eval_metric='error')

# # optimising parametres

# param_grid = {'n_estimators': np.arange(100, 1000, step=100),
#               'max_depth': [1],
#               'gamma': np.arange(0, 100),
#               'eta': np.arange(0.3, 1, step=0.1),
#               'lambda': np.arange(0, 20)
# }


# random_cv = RandomizedSearchCV(model,
#                                 param_distributions=param_grid, 
#                                 n_iter=100, 
#                                 #scoring=['precision', 'recall', 'f1'], 
#                                 cv=5, 
#                                 verbose=3, 
#                                 random_state=42)


# random_cv.fit(X_train, y_train)
# model.set_params(**random_cv.best_params_)
# model.fit(X_train, y_train)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'max_depth': 3,
    'min_child_weight': 15,
    'eta': 1,
    'gamma': 1,
    'n_estimators': 20
    }
model.set_params(**params)

df_predict = pd.DataFrame()

j=1

for train, val in skf.split(X_train, y_train):

    
    split = j
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_val = X_train.iloc[val]
    y_val = y_train.iloc[val]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_val, y_pred, labels=labels).ravel()
        
    predict = {}
    predict['set'] = 'val'
    predict['split'] = split
    predict['leadtime'] = leadtime
    predict['cor_negs'] = cor_negs
    predict['false_alarms'] = false_alarms
    predict['misses'] = misses
    predict['hits'] = hits
    predict['accuracy'] = sklearn.metrics.accuracy_score(y_val, y_pred)
    predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
    predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
    predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
    predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
    predict['precision'] = sklearn.metrics.precision_score(y_val, y_pred)
    predict['recall'] = sklearn.metrics.recall_score(y_val, y_pred)
    predict['f1'] = sklearn.metrics.f1_score(y_val, y_pred)
    df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))
    
    j+=1


ax = plot_importance(model)
ax.figure.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_importance_%sm.png"%leadtime)))


# run model


# for district in df_1month['pcode'].unique():
    
#     X_test_dist = X_test[X_test['pcode']==district].drop(columns=['pcode'])
#     y_test_dist = y_test[y_test['pcode']==district]['drought'].to_numpy()
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
# pred_district = [round(value) for value in y_pred_district]
# predict['drought'] = y_pred_district[0]
# try:
cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels).ravel()
predict = {}
predict['set'] = 'test'
predict['split'] = 0
predict['leadtime'] = leadtime
predict['cor_negs'] = cor_negs
predict['false_alarms'] = false_alarms
predict['misses'] = misses
predict['hits'] = hits
predict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
predict['precision'] = sklearn.metrics.precision_score(y_test, y_pred)
predict['recall'] = sklearn.metrics.recall_score(y_test, y_pred)
predict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))


##to do: split the test set into districts for testing


#%% 2 MONTH LEADTIME set up data for xgboost model


leadtime = leadtimes[5]

# split train-test sets

X = df_2month.drop(columns=['pcode', 'year', 'drought'])
X = StandardScaler().fit_transform(X) # normalize features
X = pd.DataFrame(data=X, columns=[1,2,9,10,11,12, 'cumul'])
y = df_2month[['drought']]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, 
                                                    stratify=y,
                                                    random_state=42)


# parametres

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(objective='binary:logistic',
                      eval_metric='error')

params = {
    'max_depth': 3,
    'min_child_weight': 10,
    'eta': 0.5,
    'gamma': 1,
    'n_estimators': 20
    }
model.set_params(**params)

# df_predict = pd.DataFrame()

j=1

for train, val in skf.split(X_train, y_train):

    
    split = j
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_val = X_train.iloc[val]
    y_val = y_train.iloc[val]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_val, y_pred, labels=labels).ravel()
        
    predict = {}
    predict['set'] = 'val'
    predict['split'] = split
    predict['leadtime'] = leadtime
    predict['cor_negs'] = cor_negs
    predict['false_alarms'] = false_alarms
    predict['misses'] = misses
    predict['hits'] = hits
    predict['accuracy'] = sklearn.metrics.accuracy_score(y_val, y_pred)
    predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
    predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
    predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
    predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
    predict['precision'] = sklearn.metrics.precision_score(y_val, y_pred)
    predict['recall'] = sklearn.metrics.recall_score(y_val, y_pred)
    predict['f1'] = sklearn.metrics.f1_score(y_val, y_pred)
    df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))
    
    j+=1


ax = plot_importance(model)
ax.figure.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_importance_%sm.png"%leadtime)))


# run model

y_pred = model.predict(X_test)

cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels).ravel()
predict = {}
predict['set'] = 'test'
predict['split'] = 0
predict['leadtime'] = leadtime
predict['cor_negs'] = cor_negs
predict['false_alarms'] = false_alarms
predict['misses'] = misses
predict['hits'] = hits
predict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
predict['precision'] = sklearn.metrics.precision_score(y_test, y_pred)
predict['recall'] = sklearn.metrics.recall_score(y_test, y_pred)
predict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))


#%% 3 MONTH LEADTIME set up data for xgboost model


leadtime = leadtimes[4]

# split train-test sets

X = df_3month.drop(columns=['pcode', 'year', 'drought'])
X = StandardScaler().fit_transform(X) # normalize features
X = pd.DataFrame(data=X, columns=[1,9,10,11,12, 'cumul'])
y = df_3month[['drought']]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, 
                                                    stratify=y,
                                                    random_state=42)


# parametres

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(objective='binary:logistic',
                      eval_metric='error')

params = {
    'max_depth': 3,
    'min_child_weight': 7,
    'eta': 0.5,
    'gamma': 1,
    'n_estimators': 20
    }
model.set_params(**params)

# df_predict = pd.DataFrame()

j=1

for train, val in skf.split(X_train, y_train):

    
    split = j
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_val = X_train.iloc[val]
    y_val = y_train.iloc[val]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_val, y_pred, labels=labels).ravel()
        
    predict = {}
    predict['set'] = 'val'
    predict['split'] = split
    predict['leadtime'] = leadtime
    predict['cor_negs'] = cor_negs
    predict['false_alarms'] = false_alarms
    predict['misses'] = misses
    predict['hits'] = hits
    predict['accuracy'] = sklearn.metrics.accuracy_score(y_val, y_pred)
    predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
    predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
    predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
    predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
    predict['precision'] = sklearn.metrics.precision_score(y_val, y_pred)
    predict['recall'] = sklearn.metrics.recall_score(y_val, y_pred)
    predict['f1'] = sklearn.metrics.f1_score(y_val, y_pred)
    df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))
    
    j+=1


ax = plot_importance(model)
ax.figure.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_importance_%sm.png"%leadtime)))


# run model

y_pred = model.predict(X_test)

cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels).ravel()
predict = {}
predict['set'] = 'test'
predict['split'] = 0
predict['leadtime'] = leadtime
predict['cor_negs'] = cor_negs
predict['false_alarms'] = false_alarms
predict['misses'] = misses
predict['hits'] = hits
predict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
predict['precision'] = sklearn.metrics.precision_score(y_test, y_pred)
predict['recall'] = sklearn.metrics.recall_score(y_test, y_pred)
predict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))


#%% 4 MONTH LEADTIME set up data for xgboost model


leadtime = leadtimes[3]

# split train-test sets

X = df_4month.drop(columns=['pcode', 'year', 'drought'])
X = StandardScaler().fit_transform(X) # normalize features
X = pd.DataFrame(data=X, columns=[9, 10, 11, 12, 'cumul'])
y = df_4month[['drought']]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, 
                                                    stratify=y,
                                                    random_state=42)


# parametres

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(objective='binary:logistic',
                      eval_metric='error')

params = {
    'max_depth': 5,
    'min_child_weight': 3,
    'eta': 0.5,
    'gamma': 1,
    'n_estimators': 20
    }
model.set_params(**params)

# df_predict = pd.DataFrame()

j=1

for train, val in skf.split(X_train, y_train):

    
    split = j
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_val = X_train.iloc[val]
    y_val = y_train.iloc[val]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_val, y_pred, labels=labels).ravel()
        
    predict = {}
    predict['set'] = 'val'
    predict['split'] = split
    predict['leadtime'] = leadtime
    predict['cor_negs'] = cor_negs
    predict['false_alarms'] = false_alarms
    predict['misses'] = misses
    predict['hits'] = hits
    predict['accuracy'] = sklearn.metrics.accuracy_score(y_val, y_pred)
    predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
    predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
    predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
    predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
    predict['precision'] = sklearn.metrics.precision_score(y_val, y_pred)
    predict['recall'] = sklearn.metrics.recall_score(y_val, y_pred)
    predict['f1'] = sklearn.metrics.f1_score(y_val, y_pred)
    df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))
    
    j+=1


ax = plot_importance(model)
ax.figure.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_importance_%sm.png"%leadtime)))


# run model

y_pred = model.predict(X_test)

cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels).ravel()
predict = {}
predict['set'] = 'test'
predict['split'] = 0
predict['leadtime'] = leadtime
predict['cor_negs'] = cor_negs
predict['false_alarms'] = false_alarms
predict['misses'] = misses
predict['hits'] = hits
predict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
predict['precision'] = sklearn.metrics.precision_score(y_test, y_pred)
predict['recall'] = sklearn.metrics.recall_score(y_test, y_pred)
predict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))



#%% 5 MONTH LEADTIME set up data for xgboost model


leadtime = leadtimes[2]

# split train-test sets

X = df_5month.drop(columns=['pcode', 'year', 'drought'])
X = StandardScaler().fit_transform(X) # normalize features
X = pd.DataFrame(data=X, columns=[9, 10, 11, 'cumul'])
y = df_5month[['drought']]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, 
                                                    stratify=y,
                                                    random_state=42)


# parametres

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(objective='binary:logistic',
                      eval_metric='error')

params = {
    'max_depth': 3,
    'min_child_weight': 2,
    'eta': 0.5,
    'gamma': 1,
    'n_estimators': 20
    }
model.set_params(**params)

# df_predict = pd.DataFrame()

j=1

for train, val in skf.split(X_train, y_train):

    
    split = j
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_val = X_train.iloc[val]
    y_val = y_train.iloc[val]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_val, y_pred, labels=labels).ravel()
        
    predict = {}
    predict['set'] = 'val'
    predict['split'] = split
    predict['leadtime'] = leadtime
    predict['cor_negs'] = cor_negs
    predict['false_alarms'] = false_alarms
    predict['misses'] = misses
    predict['hits'] = hits
    predict['accuracy'] = sklearn.metrics.accuracy_score(y_val, y_pred)
    predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
    predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
    predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
    predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
    predict['precision'] = sklearn.metrics.precision_score(y_val, y_pred)
    predict['recall'] = sklearn.metrics.recall_score(y_val, y_pred)
    predict['f1'] = sklearn.metrics.f1_score(y_val, y_pred)
    df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))
    
    j+=1


ax = plot_importance(model)
ax.figure.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_importance_%sm.png"%leadtime)))


# run model

y_pred = model.predict(X_test)

cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels).ravel()
predict = {}
predict['set'] = 'test'
predict['split'] = 0
predict['leadtime'] = leadtime
predict['cor_negs'] = cor_negs
predict['false_alarms'] = false_alarms
predict['misses'] = misses
predict['hits'] = hits
predict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
predict['precision'] = sklearn.metrics.precision_score(y_test, y_pred)
predict['recall'] = sklearn.metrics.recall_score(y_test, y_pred)
predict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))




#%% 6 MONTH LEADTIME set up data for xgboost model


leadtime = leadtimes[1]

# split train-test sets

X = df_6month.drop(columns=['pcode', 'year', 'drought'])
X = StandardScaler().fit_transform(X) # normalize features
X = pd.DataFrame(data=X, columns=[9, 10, 'cumul'])
y = df_6month[['drought']]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, 
                                                    stratify=y,
                                                    random_state=42)


# parametres

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(objective='binary:logistic',
                      eval_metric='error')

params = {
    'max_depth': 4,
    'min_child_weight': 1,
    'eta': 0.5,
    'gamma': 1,
    'n_estimators': 20
    }
model.set_params(**params)

# df_predict = pd.DataFrame()

j=1

for train, val in skf.split(X_train, y_train):

    
    split = j
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_val = X_train.iloc[val]
    y_val = y_train.iloc[val]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_val, y_pred, labels=labels).ravel()
        
    predict = {}
    predict['set'] = 'val'
    predict['split'] = split
    predict['leadtime'] = leadtime
    predict['cor_negs'] = cor_negs
    predict['false_alarms'] = false_alarms
    predict['misses'] = misses
    predict['hits'] = hits
    predict['accuracy'] = sklearn.metrics.accuracy_score(y_val, y_pred)
    predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
    predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
    predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
    predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
    predict['precision'] = sklearn.metrics.precision_score(y_val, y_pred)
    predict['recall'] = sklearn.metrics.recall_score(y_val, y_pred)
    predict['f1'] = sklearn.metrics.f1_score(y_val, y_pred)
    df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))
    
    j+=1


ax = plot_importance(model)
ax.figure.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_importance_%sm.png"%leadtime)))

# run model

y_pred = model.predict(X_test)

cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels).ravel()
predict = {}
predict['set'] = 'test'
predict['split'] = 0
predict['leadtime'] = leadtime
predict['cor_negs'] = cor_negs
predict['false_alarms'] = false_alarms
predict['misses'] = misses
predict['hits'] = hits
predict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
predict['precision'] = sklearn.metrics.precision_score(y_test, y_pred)
predict['recall'] = sklearn.metrics.recall_score(y_test, y_pred)
predict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))


#%% 7 MONTH LEADTIME set up data for xgboost model


leadtime = leadtimes[0]

# split train-test sets

X = df_7month.drop(columns=['pcode', 'year', 'drought'])
X = StandardScaler().fit_transform(X) # normalize features
X = pd.DataFrame(data=X, columns=[9])
y = df_7month[['drought']]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, 
                                                    stratify=y,
                                                    random_state=42)


# parametres

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(objective='binary:logistic',
                      eval_metric='error')

params = {
    'max_depth': 1,
    # 'min_child_weight': 1,
    # 'eta': 0.5,
    # 'gamma': 1,
    'n_estimators': 20
    }
model.set_params(**params)

# df_predict = pd.DataFrame()

j=1

for train, val in skf.split(X_train, y_train):

    
    split = j
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_val = X_train.iloc[val]
    y_val = y_train.iloc[val]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_val, y_pred, labels=labels).ravel()
        
    predict = {}
    predict['set'] = 'val'
    predict['split'] = split
    predict['leadtime'] = leadtime
    predict['cor_negs'] = cor_negs
    predict['false_alarms'] = false_alarms
    predict['misses'] = misses
    predict['hits'] = hits
    predict['accuracy'] = sklearn.metrics.accuracy_score(y_val, y_pred)
    predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
    predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
    predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
    predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
    predict['precision'] = sklearn.metrics.precision_score(y_val, y_pred)
    predict['recall'] = sklearn.metrics.recall_score(y_val, y_pred)
    predict['f1'] = sklearn.metrics.f1_score(y_val, y_pred)
    df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))
    
    j+=1


ax = plot_importance(model)
ax.figure.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_importance_%sm.png"%leadtime)))

# run model

y_pred = model.predict(X_test)

cor_negs, false_alarms, misses, hits = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels).ravel()
predict = {}
predict['set'] = 'test'
predict['split'] = 0
predict['leadtime'] = leadtime
predict['cor_negs'] = cor_negs
predict['false_alarms'] = false_alarms
predict['misses'] = misses
predict['hits'] = hits
predict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
predict['pod'] = hits / (hits + misses) if (hits + misses) != 0 else 0
predict['far'] = false_alarms / (hits + false_alarms) if (hits + false_alarms) != 0 else 0
predict['pofd'] = false_alarms / (false_alarms + cor_negs) if (false_alarms + cor_negs) != 0 else 0
predict['csi'] = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) != 0 else 0
predict['precision'] = sklearn.metrics.precision_score(y_test, y_pred)
predict['recall'] = sklearn.metrics.recall_score(y_test, y_pred)
predict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
df_predict = df_predict.append(pd.DataFrame(data=predict, index=[0]))



#%% PLOT SCORES

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10,5))
fig.tight_layout(pad=2)
# sns.scatterplot(x='leadtime', y='accuracy', hue='set', data=df_predict)
sns.scatterplot(x='leadtime', y='precision', 
                hue='set', data=df_predict, ax=ax[0])
sns.scatterplot(x='leadtime', y='recall', 
                hue='set', data=df_predict, ax=ax[1])
sns.scatterplot(x='leadtime', y='f1', 
                hue='set', data=df_predict, ax=ax[2])
ax[0].set_title('Precision')
# ax[0].yaxis.set_visible(False)
ax[0].grid()
ax[1].set_title('Recall')
# ax[1].yaxis.set_visible(False)
ax[1].grid()
ax[2].set_title('F1')
# ax[2].yaxis.set_visible(False)
ax[2].grid()
fig.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_scores1.png")))



#%% PLOT SCORES

fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15,5))
fig.tight_layout(pad=2)
# sns.scatterplot(x='leadtime', y='accuracy', hue='set', data=df_predict)
sns.scatterplot(x='leadtime', y='pod', 
                hue='set', data=df_predict, ax=ax[0])
sns.scatterplot(x='leadtime', y='far', 
                hue='set', data=df_predict, ax=ax[1])
sns.scatterplot(x='leadtime', y='pofd', 
                hue='set', data=df_predict, ax=ax[2])
sns.scatterplot(x='leadtime', y='csi', 
                hue='set', data=df_predict, ax=ax[3])
ax[0].set_title('PoD')
# ax[0].yaxis.set_visible(False)
ax[0].grid()
ax[1].set_title('FAR')
# ax[1].yaxis.set_visible(False)
ax[1].grid()
ax[2].set_title('PoFD')
# ax[2].yaxis.set_visible(False)
ax[2].grid()
ax[3].set_title('CSI')
# ax[3].yaxis.set_visible(False)
ax[3].grid()
fig.savefig(os.path.abspath(os.path.join("..", "output/zwe_m2_scores2.png")))



