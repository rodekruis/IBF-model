# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:29:00 2022

@author: pphung
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel
from sklearn import metrics as mt
import xgboost as xgb  # xGBoost
import seaborn as sns
from itertools import cycle


def get_tuned_model(xgboost_classifier, X, y):
    params = {'max_depth': np.arange(start=1, stop=7),
              'min_child_weight': np.arange(start=1, stop=21),
              'eta': np.arange(start=0.3, stop=1.1, step=0.1),
              'gamma': np.arange(start=0., stop=2.1, step=.1)
              # 'subsample': np.arange(start=0.5, stop=1., step=0.1),
              # 'alpha': np.arange(start=0., stop=20., step=.2),
              # 'lambda': np.arange(start=0., stop=20., step=.2),
              # 'colsample_bytree': np.arange(start=0.5, stop=1., step=0.1)
              }
    # TODO: START PARALLELIZING
    tuned_model = modsel.RandomizedSearchCV(
        xgboost_classifier, 
        param_distributions=params, 
        cv=5, 
        # scoring=mt.make_scorer(mt.accuracy_score),
        scoring=mt.make_scorer(mt.recall_score),
        verbose=0)
    best_model = tuned_model.fit(X, y)
    # TODO: STOP PARALLELIZING
    return best_model.best_params_


def create_scores_dictionary(y_test, pred_test):
    c_m = mt.confusion_matrix(y_test, pred_test, labels=[0,1])
    cor_negs, false_alarms, misses, hits = c_m.ravel()
    pod = hits / (hits + misses) #if (hits + misses) != 0 else 0
    far = false_alarms / (hits + false_alarms) #if (hits + false_alarms) != 0 else 0
    pofd = false_alarms / (false_alarms + cor_negs) #if (false_alarms + cor_negs) != 0 else 0
    csi = hits / (hits + false_alarms + misses) #if (hits + false_alarms + misses) != 0 else 0
    prec = mt.precision_score(y_test, pred_test, pos_label=1)
    rec = mt.recall_score(y_test, pred_test, pos_label=1)
    f1 = mt.f1_score(y_test, pred_test)
    acc = mt.accuracy_score(y_test, pred_test)
    auc = mt.roc_auc_score(y_test, pred_test)
    score_dict = {"Precision": prec, "Recall": rec, "F1": f1,
                  "Accuracy": acc, "AUC": auc,
                  'Hits': hits, 'FalseAlarms': false_alarms,
                  'Misses': misses, 'CorrectNegative': cor_negs,
                  'PoD': pod, 'FAR': far,
                  'PoFD': pofd, 'CSI': csi
                  }
    return score_dict

def calculate_dryspell(precip_obs, days_rolling_cumul):
    '''
    Function to calculate dryspell by definition below based on n-day rolling
    cumulative sum of rainfall per district.
    Input is a dataframe of rainfall. Each row is a daily rainfall of a district.
    
    Dry spell: At least 14 consecutive days with a cumulative rainfall of 
    no more than 2 mm during the rainy season in an ADMIN2 region. 
    This definition was provided by the World Food Programme (WFP) who also 
    shared impact survey data (MVAC).
    '''
    
    #calculate 10-day rolling cumulative rainfall per admin
    precip_obs['rolling_cumul'] = precip_obs.groupby('pcode')['rain'].\
        rolling(days_rolling_cumul).sum().reset_index(0,drop=True)
    
    #dry spell if 14-day cumulative rainfall below 2mm
    precip_obs['dryspell'] = np.where(precip_obs['rolling_cumul'] <= 2, 1, 0)
    
    #count "dryspell" date per month per admin
    precip_dryspell = precip_obs.groupby(['pcode', 'monthyear'])['dryspell'].\
        sum().reset_index()
    precip_cumul = precip_obs.groupby(['pcode', 'monthyear'])['rain'].\
        sum().reset_index()
    precip_cumul = precip_cumul.rename(columns={'rain': 'p_month_cumul'})
    precip_dryspell = precip_dryspell.merge(precip_cumul, on=['pcode', 'monthyear'])
    
    # calculate dryspell
    precip_dryspell['year_original'] = pd.to_datetime(precip_dryspell['monthyear']).\
        dt.strftime('%Y').astype(int)
    precip_dryspell['month_original'] = pd.to_datetime(precip_dryspell['monthyear']).\
        dt.strftime('%m').astype(int)
    # precip_dryspell['year'] = precip_dryspell['year_original']
    precip_dryspell['date'] = "01-01-" + precip_dryspell['year_original'].astype(str)
    precip_dryspell['date'] = pd.to_datetime(precip_dryspell['date']).\
        dt.strftime('%Y-%m-%d')
    precip_dryspell['year'] = np.where(precip_dryspell['month_original']>=7, 
                                       precip_dryspell['year_original']+1, 
                                       precip_dryspell['year_original']+0)
    precip_dryspell = precip_dryspell.drop(columns='monthyear')
    
    # pivot dryspell
    precip_dryspell_final = pd.pivot_table(precip_dryspell,
                                           values = ['dryspell', 'p_month_cumul'],
                                           index = ['pcode','year'],
                                           columns = 'month_original').reset_index()

    # create a list of the new column names in the right order
    new_cols=[('{1}_{0}'.format(*tup)) for tup in precip_dryspell_final.columns]
    
    # assign it to the dataframe (assuming you named it pivoted
    precip_dryspell_final.columns = new_cols
    precip_dryspell_final = precip_dryspell_final.\
        rename(columns={'_pcode': 'pcode',
                        '_year': 'year'})
        
    return(precip_dryspell_final)

#%%

# Load Data
path = os.getcwd()
# path = os.path.abspath(os.path.join(path,'..'))

#load shapefile admin
zwe2_shp = gpd.read_file(os.path.abspath(
    os.path.join("../..", "Model (ENSO)/M1/shape/zwe_admbnda_adm2_zimstat_ocha_20180911.shp")))
zwe2_shp = zwe2_shp[["ADM1_PCODE", "ADM2_PCODE", "ADM0_EN"]]
regions = zwe2_shp['ADM1_PCODE'].unique().tolist()

#%%

# load precipication data
precip_obs = pd.read_csv(os.path.abspath(
    os.path.join("../../..", "3. Data - Hazard exposure, vulnerability/zwe_rain/zwe_daily_rainfall.csv"))) 
precip_obs['date'] = pd.to_datetime(precip_obs['date']).dt.strftime('%Y-%m-%d')
precip_obs['monthyear'] = pd.to_datetime(precip_obs['date']).dt.strftime('%Y-%m')

# #calculate rolling cumulative rainfall per month per admin
# precip_obs['roll_cumul_month'] = precip_obs.groupby(['pcode', 'monthyear']).cumsum()#.reset_index()


# load ENSO data
enso_table = pd.read_csv(
    os.path.abspath(os.path.join("../..", "Model (ENSO)/M1/data/enso_table_new.csv"))).\
    drop(columns='Unnamed: 0')
enso_table = enso_table.rename(columns={'YR': 'Year'})
enso_table['Year'] = enso_table['Year'].astype(int)
# enso_table['Year'] = "01-01-" + enso_table['Year'].astype(str)
# enso_table['Year'] = pd.to_datetime(enso_table['Year']).\
#     dt.strftime('%Y-%m-%d')
enso_table[['NDJ', 'DJF', 'JFM']] = enso_table[
    ['NDJ', 'DJF', 'JFM']].shift(0)
# enso_table['JJA'] = enso_table['JJA'].shift(1)
# enso_table['JAS'] = enso_table['JAS'].shift(1)
# enso_table['ASO'] = enso_table['ASO'].shift(1)
# enso_table['SON'] = enso_table['SON'].shift(1)
# enso_table['OND'] = enso_table['OND'].shift(1)
# enso_table['NDJ'] = enso_table['NDJ'].shift(1)
enso_table.dropna(inplace=True)
enso_table = enso_table.drop(columns=['FMA', 'MAM', 'AMJ', 'MJJ', 'JJA'])


# load vci
# vci_thresholds = [0.25, 0.45, 0.65, 1]
# vci_threshold = vci_thresholds[1]*100
months = cycle(np.arange(1,13))
years = cycle([year for year in np.arange(1982, 2021) 
               for _ in range(12)])

vci_obs = pd.read_pickle(os.path.abspath(
    os.path.join("..", "VCI_for_510/observations_monthly.pickle")))
vci_obs = pd.DataFrame.from_dict(vci_obs)

# add date month year to dataframe
vci_obs['month'] = 0
vci_obs['year_original'] = 0
vci_obs['month'][:5] = np.arange(8,13)
vci_obs['month'][5:] = [next(iter(months))
                        for month in range(len(vci_obs)-5)]
vci_obs['year_original'][:5] = 1981
vci_obs['year_original'][5:] = [next(iter(years)) 
                       for year in range(len(vci_obs)-5)]

vci_obs['year'] = np.where(vci_obs['month']>=7, 
                           vci_obs['year_original']+1, 
                           vci_obs['year_original']+0)
vci_obs['month_txt'] = vci_obs['month'].astype(str) + '_vci_obs'
vci_obs.drop(columns=['year_original', 'month'], inplace=True)


# pivot vci_obs
vci_obs_melted = vci_obs.melt(id_vars=['year', 'month_txt'],
                              var_name='pcode',
                              value_name='vci_obs')
vci_obs_pivotted = pd.pivot_table(vci_obs_melted,
                                  values='vci_obs',
                                  index=['pcode','year'],
                                  columns='month_txt').reset_index()
vci_obs_pivotted = vci_obs_pivotted.merge(zwe2_shp.reset_index(level=0),
                                          how='left', 
                                          left_on='pcode',
                                          right_on='index')
vci_obs_pivotted = vci_obs_pivotted.drop(columns=['pcode', 'ADM1_PCODE', 'index'])
vci_obs_final = vci_obs_pivotted[[#'7_vci_obs', '8_vci_obs', 
                                  '9_vci_obs',
                                  '10_vci_obs', '11_vci_obs', '12_vci_obs',
                                  '1_vci_obs', '2_vci_obs', '3_vci_obs',
                                  'year', 'ADM2_PCODE']]

#load impact data
impact = pd.read_csv(os.path.abspath(
    os.path.join("../..", "Model (ENSO)/M1/data/zwe_crop_yield_anomaly.csv"))).\
    drop(columns='Unnamed: 0')
impact['date'] = pd.to_datetime(impact['date']).dt.strftime('%Y-%m-%d')
impact['year'] = pd.to_datetime(impact['date']).dt.strftime('%Y').astype(int)


rolling_cumul_days = [10, 14, 21, 31, 41]
leadtimes_val = list(i for i in np.arange(5, -1, -1))
parameters_df = pd.DataFrame()
skillscores_df = pd.DataFrame()

# process rainfall and dryspell
for days in rolling_cumul_days:
    precip_dryspell = calculate_dryspell(precip_obs, days)
    precip_dryspell = precip_dryspell.drop(columns=[
        '4_p_month_cumul', '5_p_month_cumul', '6_p_month_cumul', 
        '7_p_month_cumul', '8_p_month_cumul', '4_dryspell', '5_dryspell', 
        '6_dryspell', '7_dryspell', '8_dryspell']).dropna()
    
    # merge dryspell and impact
    DF = pd.merge(precip_dryspell, impact, 
                  how='left',
                  left_on=['pcode', 'year'],
                  right_on=['ADM2_PCODE', 'year']).drop(columns='ADM2_PCODE')
    DF['drought'] = np.where(DF['anomaly']=='Yes', 1, 0)
    # DF['dec_jan_feb_mar'] = DF[[12, 1, 2, 3]].sum(axis=1)
    
    # merge dryspell and impact
    DF = DF.merge(vci_obs_final, 
                  how='left',
                  left_on=['pcode', 'year'],
                  right_on=['ADM2_PCODE', 'year'])
    DF['ADM1_PCODE'] = DF['pcode'].str[0:4]
    DF.drop(columns = ['ADM2_PCODE', 'remainder', 'anomaly', 'pcode', 'date'], 
            inplace=True)
    
    # # correlation of dryspell and impact
    # # DF['dryspell_14_bin'] = np.where(DF['dryspell_14']>0, 1, 0)
    # for i in np.arange(1,13):
    #     print(DF[[i, 'drought']].corr())
    
    # merge with enso data
    DF = DF.merge(enso_table, how="left", 
                  left_on="year", right_on="Year").\
        drop(columns=['Year', 'year'])
    
    
    # # axtract features based on lead time
    # df_6month = DF.copy()
    # df_6month.drop(
    #     ['ASO', 'SON', 'OND', 'NDJ', 'DJF', 'JFM', 
    #      '10_p_month_cumul', '11_p_month_cumul', '12_p_month_cumul', 
    #      '1_p_month_cumul', '2_p_month_cumul', '3_p_month_cumul',
    #      '10_dryspell', '11_dryspell', '12_dryspell', '1_dryspell', 
    #      '2_dryspell', '3_dryspell'], axis=1, inplace=True)
    
    df_5month = DF.copy()
    df_5month.drop(
        ['SON', 'OND', 'NDJ', 'DJF', 'JFM', 
         '11_p_month_cumul', '12_p_month_cumul', '1_p_month_cumul', 
         '2_p_month_cumul', '3_p_month_cumul',
         '11_dryspell', '12_dryspell', '1_dryspell', '2_dryspell', 
         '3_dryspell',
         '11_vci_obs', '12_vci_obs', '1_vci_obs', '2_vci_obs', 
         '3_vci_obs'], axis=1, inplace=True)
    df_5month['p_cumul'] = df_5month[['9_p_month_cumul', '10_p_month_cumul']].sum(axis=1)
    df_5month['vci_avg'] = df_5month[['9_vci_obs', '10_vci_obs']].mean(axis=1)
    
    df_4month = DF.copy()
    df_4month.drop(
        ['OND', 'NDJ', 'DJF', 'JFM', 
         '12_p_month_cumul', '1_p_month_cumul', '2_p_month_cumul', 
         '3_p_month_cumul',
         '12_dryspell', '1_dryspell', '2_dryspell', '3_dryspell',
         '12_vci_obs', '1_vci_obs', '2_vci_obs', '3_vci_obs'], 
        axis=1, inplace=True)
    df_4month['p_cumul'] = df_4month[
        ['9_p_month_cumul', '10_p_month_cumul', '11_p_month_cumul']].sum(axis=1)
    df_4month['vci_avg'] = df_4month[['9_vci_obs', '10_vci_obs', '11_vci_obs']].mean(axis=1)
    
    df_3month = DF.copy()
    df_3month.drop(
        ['NDJ', 'DJF', 'JFM', 
         '1_p_month_cumul', '2_p_month_cumul', '3_p_month_cumul',
         '1_dryspell', '2_dryspell', '3_dryspell', 
         '1_vci_obs', '2_vci_obs', '3_vci_obs'], axis=1, inplace=True)
    df_3month['p_cumul'] = df_3month[
        ['9_p_month_cumul', '10_p_month_cumul', '11_p_month_cumul',
         '12_p_month_cumul']].sum(axis=1)
    df_3month['vci_avg'] = df_3month[['9_vci_obs', '10_vci_obs', '11_vci_obs',
                                      '12_vci_obs']].mean(axis=1)
    
    df_2month = DF.copy()
    df_2month.drop(
        ['DJF', 'JFM', 
         '2_p_month_cumul', '3_p_month_cumul',
         '2_dryspell', '3_dryspell',
         '2_vci_obs', '3_vci_obs'], axis=1, inplace=True)
    df_2month['p_cumul'] = df_2month[
        ['9_p_month_cumul', '10_p_month_cumul', '11_p_month_cumul', 
         '12_p_month_cumul', '1_p_month_cumul']].sum(axis=1)
    df_2month['vci_avg'] = df_2month[['9_vci_obs', '10_vci_obs', '11_vci_obs',
                                      '12_vci_obs', '1_vci_obs']].mean(axis=1)
    cols = ['JAS', 'ASO', 'SON', 'OND', 'NDJ',
            '9_p_month_cumul', '10_p_month_cumul', '11_p_month_cumul', 
            '12_p_month_cumul', '1_p_month_cumul', 
            '9_dryspell', '10_dryspell', '11_dryspell', '12_dryspell', 
            '1_dryspell', 
            '9_vci_obs', '10_vci_obs', '11_vci_obs', '12_vci_obs', 
            '1_vci_obs',
            'p_cumul', 'vci_avg', 'ADM1_PCODE', 'drought'] # desired order of columns
    df_2month = df_2month[cols] # reorder columns
    
    df_1month = DF.copy()
    df_1month.drop(
        ['JFM', '3_p_month_cumul', '3_dryspell', '3_vci_obs'], 
        axis=1, inplace=True)
    df_1month['p_cumul'] = df_1month[
        ['9_p_month_cumul', '10_p_month_cumul', '11_p_month_cumul', 
         '12_p_month_cumul', '1_p_month_cumul', '2_p_month_cumul']].sum(axis=1)
    df_1month['vci_avg'] = df_1month[['9_vci_obs', '10_vci_obs', '11_vci_obs',
                                      '12_vci_obs', '1_vci_obs', '2_vci_obs']].\
        mean(axis=1)
    cols = ['JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF',
            '9_p_month_cumul', '10_p_month_cumul', '11_p_month_cumul', 
            '12_p_month_cumul', '1_p_month_cumul', '2_p_month_cumul', 
            '9_dryspell', '10_dryspell', '11_dryspell', '12_dryspell', 
            '1_dryspell', '2_dryspell',
            '9_vci_obs', '10_vci_obs', '11_vci_obs', '12_vci_obs', 
            '1_vci_obs', '2_vci_obs',
            'p_cumul', 'vci_avg', 'ADM1_PCODE', 'drought'] # desired order of columns
    df_1month = df_1month[cols] # reorder columns
    
    df_0month = DF.copy()
    df_0month['p_cumul'] = df_0month[
        ['9_p_month_cumul', '10_p_month_cumul', '11_p_month_cumul', 
         '12_p_month_cumul', '1_p_month_cumul', '2_p_month_cumul', 
         '3_p_month_cumul']].sum(axis=1)
    df_0month['vci_avg'] = df_0month[['9_vci_obs', '10_vci_obs', '11_vci_obs',
                                      '12_vci_obs', '1_vci_obs', '2_vci_obs', 
                                      '3_vci_obs']].mean(axis=1)
    cols = ['JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF', 'JFM',
            '9_p_month_cumul', '10_p_month_cumul', '11_p_month_cumul', 
            '12_p_month_cumul', '1_p_month_cumul', '2_p_month_cumul',  
            '3_p_month_cumul', 
            '9_dryspell', '10_dryspell', '11_dryspell', 
            '12_dryspell', '1_dryspell', '2_dryspell', '3_dryspell',
            '9_vci_obs', '10_vci_obs', '11_vci_obs', '12_vci_obs', 
            '1_vci_obs', '2_vci_obs', '3_vci_obs',
            'p_cumul', 'vci_avg', 'ADM1_PCODE', 'drought'] # desired order of columns
    df_0month = df_0month[cols] # reorder columns

    # skill analysis with xgboost
    # parameters_df = pd.DataFrame()
    # skillscores_df = pd.DataFrame()
    # leadtimes = []
    leadtimes = [df_5month, df_4month, df_3month, df_2month, 
                 df_1month, df_0month]
    i = 0
    for leadtime in leadtimes:
        df = leadtime.copy()
        predictors = df.drop(columns=['drought']).copy()
        target = df[['drought', 'ADM1_PCODE']].copy()
        X_train, X_test, y_train, y_test = \
            modsel.train_test_split(predictors, target,
                                    test_size=0.3,
                                    stratify=target)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100, objective='binary:logistic', seed=123)
        
        
        X_train = X_train.drop(columns='ADM1_PCODE')
        y_train = y_train.drop(columns='ADM1_PCODE')
        best_params = get_tuned_model(xgb_clf, X_train, y_train)
        print("region: {} \t n_months_lead:{} \nbest_params :{}".\
              format('all', leadtimes_val[i], best_params))
        parameters = pd.DataFrame(data=best_params, index=[0])
        parameters['dryspell_dur'] = days
        parameters['region'] = 'all'#region
        parameters['leadtime'] = leadtimes_val[i]
        # print("PARAMETERS: {}".format(parameters))
        if i == 0:
            parameters_df = parameters.copy()
        else:
            parameters_df = pd.concat(
                [parameters_df, parameters], ignore_index=True)
        # print("PARAMETERS_DF: {}".format(parameters_df))
        # what do we do with these parameters in a dataframe?
    
        xgb_clf_best = xgb.XGBClassifier(n_estimators=100, 
                                         objective='binary:logistic', 
                                         seed=123)
        xgb_clf_best.set_params(**best_params)
        xgb_clf_best.fit(X_train, y_train)
        
        y_pred = xgb_clf_best.predict(X_train)
        # tN, fP, fN, tP = mt.confusion_matrix(y_train, y_pred, labels=[0,1]).ravel()
        score_dict = create_scores_dictionary(y_train, y_pred)
        score_dict['set'] = 'val'
        # score_dict['split'] = split
        score_dict['dryspell_dur'] = days
        score_dict['leadtime'] = leadtimes_val[i]
        score_dict['region'] = 'all'
        skillscores_df = skillscores_df.append(pd.DataFrame(data=score_dict, index=[0]))
        
        # plot feature importance
        feature_importance = pd.DataFrame(
            data=[list(X_train), 
                  xgb_clf_best.feature_importances_.tolist()]).T
        feature_importance = feature_importance.rename(
            columns={0: 'feature', 1: 'importance'})
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10,5))
        fig.tight_layout(pad=2)
        ax.set_title('Feature importance at {0}-month lead time, {1}-day dry spells'.format(leadtimes_val[i], days))
        # ax[0].yaxis.set_visible(False)
        ax.grid()
        ax.set_axisbelow(True)
        sns.barplot(x='feature', y='importance',
                    data=feature_importance, ax=ax)
        ax.tick_params(axis='x', rotation=30)
        plt.subplots_adjust(bottom=0.2)
        # fig.savefig(os.path.abspath(
        #     os.path.join("..", "output/maize(enso+rain+dryspell)/zwe_m2ensoraindryspell_features_leadtime{0}_dryspell{1}.png".format(leadtimes_val[i], days))))
    
        model_name = "zwe_m3_crop_{0}_model.json".format(leadtimes_val[i])
        xgb_clf_best.save_model(os.path.abspath(os.path.join("..", "results/maize(enso+rain+dryspell+vci)/" + model_name)))
        # print(model_name)
        
        for region in X_test['ADM1_PCODE'].unique():
            X_test_dist = X_test[X_test['ADM1_PCODE']==region].drop(columns=['ADM1_PCODE'])
            y_test_dist = y_test[y_test['ADM1_PCODE']==region]['drought'].to_numpy()
            
            
            pred_test = xgb_clf_best.predict(X_test_dist)
        
            # TODO:
            # check if this is in agreement with precision and recall
            # definitions of positive case
            # tN, fP, fN, tP = mt.confusion_matrix(y_test, pred_test, labels=[0,1]).ravel()
            score_dict = create_scores_dictionary(y_test_dist, pred_test)
            score_dict['set'] = 'test'
            # score_dict['split'] = split
            score_dict['dryspell_dur'] = days
            score_dict['leadtime'] = leadtimes_val[i]
            score_dict['region'] = region
            # df_score_test = pd.DataFrame(list(score_dict.items()),
            #                         columns=["Metric", "score_test"])
            # print(df_score_test)
            
            skillscores_df = skillscores_df.append(pd.DataFrame(data=score_dict, index=[0]))
    
        i = i + 1
    # parameters_df.to_csv(path="/results/output/parameters_df_{}.csv".format(region))
    

# save skill scores as a csv file
# skillscores_df.to_csv(os.path.abspath(
#     os.path.join("..", "output/maize(enso+rain+dryspell+vci)/zwe_skillscore_ensoraindryspellvci.csv")), index=False)

#%% PLOT SCORES

skillscores_df = pd.read_csv(os.path.abspath(
    os.path.join("..", "output/maize(enso+rain+dryspell+vci)/zwe_skillscore_ensoraindryspellvci.csv")))

# fill NaN with -0.1 for plotting purpose
skillscores_df = skillscores_df.fillna(-0.1)

for region in regions:
    skillscores_df_district = skillscores_df[skillscores_df['region']==region]
    skillscores_df_district = skillscores_df_district[skillscores_df_district['dryspell_dur']==14]
    # Precision, Recall, F1
    # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
    # fig.tight_layout(pad=2)
    # # sns.scatterplot(x='leadtime', y='accuracy', hue='region', data=df_predict)
    # sns.scatterplot(x='leadtime', y='Precision', 
    #                 hue='region', data=skillscores_df_district, ax=ax[0])
    # sns.scatterplot(x='leadtime', y='Recall', 
    #                 hue='region', data=skillscores_df_district, ax=ax[1])
    # sns.scatterplot(x='leadtime', y='F1', 
    #                 hue='region', data=skillscores_df_district, ax=ax[2])
    # ax[0].set_title('Precision')
    # # ax[0].yaxis.set_visible(False)
    # ax[0].grid()
    # ax[1].set_title('Recall')
    # # ax[1].yaxis.set_visible(False)
    # ax[1].grid()
    # ax[2].set_title('F1')
    # # ax[2].yaxis.set_visible(False)
    # ax[2].grid()
    # # plt.setp(ax, ylim=(0,1))
    # # fig.savefig(os.path.abspath(os.path.join("..", "output/maize/zwe_m2_scores1_maize.png")))
    
    
    
    # PoD, FAR, PoFD, CSI
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15,5))
    fig.suptitle(region, fontsize=16)
    fig.tight_layout(pad=2)
    ax[0].set_title('PoD')
    # ax[0].yaxis.set_visible(False)
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[1].set_title('FAR')
    # ax[1].yaxis.set_visible(False)
    ax[1].grid()
    ax[1].set_axisbelow(True)
    # ax[2].set_title('PoFD')
    # # ax[2].yaxis.set_visible(False)
    # ax[2].grid()
    # ax[3].set_title('CSI')
    # # ax[3].yaxis.set_visible(False)
    # ax[3].grid()
    sns.scatterplot(x='leadtime', y='PoD', palette="#00214D", s=100,
                data=skillscores_df_district, ax=ax[0])
    sns.scatterplot(x='leadtime', y='FAR', palette="#00214D", s=100,
                data=skillscores_df_district, ax=ax[1])
    # sns.scatterplot(x='leadtime', y='PoFD', 
    #                 hue='dryspell_dur', data=skillscores_df_district, ax=ax[2])
    # sns.scatterplot(x='leadtime', y='CSI', 
    #                 hue='dryspell_dur', data=skillscores_df_district, ax=ax[3])
    plt.setp(ax, ylim=(-0.05,1.05))
    
    # fig.savefig(os.path.abspath(os.path.join("..", "output/maize(enso+rain+dryspell+vci)/for_sander/zwe_m3_scores2_%s.png"%region)))
