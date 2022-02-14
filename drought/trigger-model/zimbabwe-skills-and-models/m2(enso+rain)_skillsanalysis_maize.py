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
        xgboost_classifier, param_distributions=params, cv=5, scoring=mt.make_scorer(mt.accuracy_score),
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
    score_dict = {"Precision": prec, "Recall": rec, "F1": f1,
                  "Accuracy": acc, 
                  'Hits': hits, 'FalseAlarms': false_alarms,
                  'Misses': misses, 'CorrectNegative': cor_negs,
                  'PoD': pod, 'FAR': far,
                  'PoFD': pofd, 'CSI': csi
                  }
    return score_dict


if __name__ == "__main__":

    import os
    import numpy as np
    import pandas as pd  # dplyr
    import shapefile
    import geopandas as gpd
    import xgboost as xgb  # xGBoost
    import matplotlib.pyplot as plt  # ggplot2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # TODO: check parallel programming
    from multiprocessing import Pool  #
    # import ipyparallel
    import joblib
    # import pycaret as pycar  #caret
    from sklearn import model_selection as modsel
    from sklearn import metrics as mt
    from statsmodels.stats.contingency_tables import mcnemar
    import seaborn as sns

    # Load Data
    path = os.getcwd()
    # path = os.path.abspath(os.path.join(path,'..'))

    zwe1 = shapefile.Reader(os.path.abspath(os.path.join("../..", "Model (ENSO)/M1/shape/zwe_admbnda_adm1_zimstat_ocha_20180911.shp")))
    fields1 = [x[0] for x in zwe1.fields][1:]
    records1 = [y[:] for y in zwe1.records()]
    shps1 = [s.points for s in zwe1.shapes()]
    zwe_adm1 = pd.DataFrame(columns=fields1, data=records1)
    zwe_adm1 = zwe_adm1.assign(geometry=shps1)
    # print("ZWE_ADM1")
    # print(zwe_adm1)
    zwe_adm1 = zwe_adm1[["ADM1_PCODE", "ADM0_EN"]]
    
    zwe2_shp = gpd.read_file(os.path.abspath(os.path.join("../..", "Model (ENSO)/M1/shape/zwe_admbnda_adm2_zimstat_ocha_20180911.shp")))
    # zwe2 = shapefile.Reader(os.path.abspath(os.path.join("../..", "Model (ENSO)/M1/shape/zwe_admbnda_adm2_zimstat_ocha_20180911.shp")))
    zwe2 = shapefile.Reader(os.path.abspath(os.path.join("..", "input/zwe_lhz/zwe_maize_areas.shp")))
    fields2 = [x[0] for x in zwe2.fields][1:]
    records2 = [y[:] for y in zwe2.records()]
    shps2 = [s.points for s in zwe2.shapes()]
    zwe_adm2 = pd.DataFrame(columns=fields2, data=records2)
    zwe_adm2 = zwe_adm2.assign(geometry=shps2)
    zwe_adm2 = zwe_adm2[["ADM1_PCODE", "ADM2_PCODE", "ADM0_EN"]]
    # print("ZWE_ADM2")
    # print(zwe_adm2)
    
    enso_table = pd.read_csv(os.path.abspath(os.path.join("../..", "Model (ENSO)/M1/data/enso_table_new.csv"))).drop(columns='Unnamed: 0')
    enso_table = enso_table.rename(columns={'YR': 'Year'})
    enso_table['Year'] = enso_table['Year'].astype(int)
    enso_table['Year'] = "01-01-" + enso_table['Year'].astype(str)
    enso_table['Year'] = pd.to_datetime(enso_table['Year']).dt.strftime('%Y-%m-%d')
    enso_table[['NDJ', 'DJF', 'JFM']] = enso_table[['NDJ', 'DJF', 'JFM']].shift(0)
    # enso_table['JJA'] = enso_table['JJA'].shift(1)
    # enso_table['JAS'] = enso_table['JAS'].shift(1)
    # enso_table['ASO'] = enso_table['ASO'].shift(1)
    # enso_table['SON'] = enso_table['SON'].shift(1)
    # enso_table['OND'] = enso_table['OND'].shift(1)
    # enso_table['NDJ'] = enso_table['NDJ'].shift(1)
    enso_table.dropna(inplace=True)
    # print("ENSO!!!!!")
    # print(enso_table)
    
    precip_obs = pd.read_csv(os.path.abspath(os.path.join("../../..", "3. Data - Hazard exposure, vulnerability/zwe_rain/zwe_monthcumulative_rainfall.csv")))    
    # precip_obs['date'] = pd.to_datetime(precip_obs["year"].astype(str) + '-' + precip_obs["month"].astype(str), format='%Y-%m')
    precip_obs = precip_obs.rename(columns={'year':'year_original'})
    precip_obs['year'] = precip_obs['year_original']
    precip_obs['date'] = "01-01-" + precip_obs['year_original'].astype(str)
    precip_obs['date'] = pd.to_datetime(precip_obs['date']).dt.strftime('%Y-%m-%d')
    precip_obs['year'] = np.where(precip_obs['month']>=7, precip_obs['year_original']+1, precip_obs['year_original']+0)
    precip_obs = pd.pivot_table(precip_obs, values = 'pcumul', index=['pcode','date'], columns = 'month').reset_index()
    precip_obs = precip_obs[precip_obs['date'] != '1999-01-01']
    # print("Precipitation!!!!!")
    # print(precip_obs)

    impact = pd.read_csv(os.path.abspath(os.path.join("../..", "Model (ENSO)/M1/data/zwe_crop_yield_anomaly.csv")))

    impact = pd.merge(impact, zwe_adm2, on='ADM2_PCODE')
    # not needed to drop geometry because it is not there
    ## impact.drop(labels='geometry', axis=1, inplace=True)
    # not needed to makeit a datetime, better to merge on strings
    ## impact["date"] = pd.to_datetime(impact["date"])
    # print("IMPACT!!!!")
    # print(impact)
    adm2df = pd.DataFrame(impact['ADM2_PCODE'].unique(), columns=['ADM2_PCODE'])

    datelist = pd.date_range(start="2000-01-01", periods=38, freq="YS").tolist()
    datedf = pd.DataFrame(datelist, columns=['date'])
    datedf['date'] = datedf['date'].dt.strftime('%Y-%m-%d')

    filled = datedf.merge(adm2df, how='cross')
    filled["date"] = filled["date"].astype(str)
    # print("FILLED!!!!")
    # print(filled)
    DF = filled.merge(impact, how='left', on=['date', 'ADM2_PCODE'])
    print(DF)
    DF['ADM1_PCODE'] = DF['ADM2_PCODE'].str[0:4]
    DF = DF.merge(enso_table, how="left", left_on="date", right_on="Year")
    DF = DF.merge(precip_obs, how="left", on="date")
    DF['drought'] = np.where(DF['anomaly'] == "Yes", 1, 0)
    DF['drought_binary'] = np.where(DF['anomaly'] == "Yes", 1, 0)
    DF.drop(['Unnamed: 0', 'anomaly', 'remainder', 'date', 'ADM2_PCODE', 'pcode',
             'ADM0_EN', 'drought'], axis=1, inplace=True)
    DF['drought_binary'].fillna(0)
    DF[9].replace('', np.nan, inplace=True)
    DF.dropna(subset=[9], inplace=True)
    # print("DF!!!!!!")
    # print(DF)

    # # Optional Renaming (it is only to lower case)
    # impact.rename(columns={'ADM2_PCODE': 'adm2_pcode'}, inplace=True)
    #
    leadtimes_val = list(i for i in np.arange(6, -1, -1))
    regions = zwe_adm2['ADM1_PCODE'].unique().tolist()
    parameters_df = pd.DataFrame()
    skillscores_df = pd.DataFrame()
    leadtimes = []
    

    df_6month = DF.copy()
    df_6month.drop(
        ['Year', 'ASO', 'SON', 'OND', 'NDJ', 'DJF', 'JFM', 10, 11, 12, 1, 2, 3], axis=1, inplace=True)
    df_5month = DF.copy()
    df_5month.drop(
        ['Year', 'SON', 'OND', 'NDJ', 'DJF', 'JFM', 11, 12, 1, 2, 3], axis=1, inplace=True)
    df_5month['cumul'] = df_5month[[9, 10]].sum(axis=1)
    df_4month = DF.copy()
    df_4month.drop(
        ['Year', 'OND', 'NDJ', 'DJF', 'JFM', 12, 1, 2, 3], axis=1, inplace=True)
    df_4month['cumul'] = df_4month[[9, 10, 11]].sum(axis=1)
    df_3month = DF.copy()
    df_3month.drop(
        ['Year', 'NDJ', 'DJF', 'JFM', 1, 2, 3], axis=1, inplace=True)
    df_3month['cumul'] = df_3month[[9, 10, 11, 12]].sum(axis=1)
    df_2month = DF.copy()
    df_2month.drop(['Year', 'DJF', 'JFM', 2, 3], axis=1, inplace=True)
    df_2month['cumul'] = df_2month[[9, 10, 11, 12, 1]].sum(axis=1)
    df_1month = DF.copy()
    df_1month.drop(['Year', 'JFM', 3], axis=1, inplace=True)
    df_1month['cumul'] = df_1month[[9, 10, 11, 12, 1, 2]].sum(axis=1)
    df_0month = DF.copy()
    df_0month.drop(['Year'], axis=1, inplace=True)
    df_0month['cumul'] = df_0month[[9, 10, 11, 12, 1, 2, 3]].sum(axis=1)

    leadtimes = [df_6month, df_5month, df_4month, df_3month, df_2month, df_1month, df_0month]
    i = 0
    for leadtime in leadtimes:
        df = leadtime.copy()
        predictors = df.drop(columns=['drought_binary']).copy()
        target = df[['drought_binary', 'ADM1_PCODE']].copy()
        X_train, X_test, y_train, y_test = \
            modsel.train_test_split(predictors, target, test_size=0.2)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100, objective='binary:logistic', seed=123)
        # REMEMBER eval_metric in python is associated with the train/fit methods,
        # not with the creation of the model
        
        X_train = X_train.drop(columns='ADM1_PCODE')
        y_train = y_train.drop(columns='ADM1_PCODE')
        best_params = get_tuned_model(xgb_clf, X_train, y_train)
        print("region: {} \t n_months_lead:{} \nbest_params :{}".format('all', leadtimes_val[i], best_params))
        parameters = pd.DataFrame(data=best_params, index=[0])
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

        xgb_clf_best = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', seed=123)
        xgb_clf_best.set_params(**best_params)
        xgb_clf_best.fit(X_train, y_train)
        
        y_pred = xgb_clf_best.predict(X_train)
        # tN, fP, fN, tP = mt.confusion_matrix(y_train, y_pred, labels=[0,1]).ravel()
        score_dict = create_scores_dictionary(y_train, y_pred)
        score_dict['set'] = 'val'
        # score_dict['split'] = split
        score_dict['leadtime'] = leadtimes_val[i]
        score_dict['region'] = 'all'
        skillscores_df = skillscores_df.append(pd.DataFrame(data=score_dict, index=[0]))

        model_name = "zwe_m2_maize_" \
                          + (leadtimes_val[i]).astype(str) + "_model.json"
        xgb_clf_best.save_model(os.path.abspath(os.path.join("..", "results/maize(enso+rain)/zwe_m2_crop_" \
                                                              + (leadtimes_val[i]).astype(str) + "_model.json")))
        # print(model_name)
        
        for region in X_test['ADM1_PCODE'].unique():
            X_test_dist = X_test[X_test['ADM1_PCODE']==region].drop(columns=['ADM1_PCODE'])
            y_test_dist = y_test[y_test['ADM1_PCODE']==region]['drought_binary'].to_numpy()
            
            
            pred_test = xgb_clf_best.predict(X_test_dist)
    
            # TODO:
            # check if this is in agreement with precision and recall
            # definitions of positive case
            # tN, fP, fN, tP = mt.confusion_matrix(y_test, pred_test, labels=[0,1]).ravel()
            score_dict = create_scores_dictionary(y_test_dist, pred_test)
            score_dict['set'] = 'test'
            # score_dict['split'] = split
            score_dict['leadtime'] = leadtimes_val[i]
            score_dict['region'] = region
            # df_score_test = pd.DataFrame(list(score_dict.items()),
            #                         columns=["Metric", "score_test"])
            # print(df_score_test)
            
            skillscores_df = skillscores_df.append(pd.DataFrame(data=score_dict, index=[0]))

        i = i + 1
    # parameters_df.to_csv(path="/results/output/parameters_df_{}.csv".format(region))


# THE END

    #%% PLOT SCORES


    for region in regions:
        # Precision, Recall, F1
        
        # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
        # fig.tight_layout(pad=2)
        # # sns.scatterplot(x='leadtime', y='accuracy', hue='region', data=df_predict)
        # sns.scatterplot(x='leadtime', y='Precision', 
        #                 hue='region', data=skillscores_df[skillscores_df['region']==region], ax=ax[0])
        # sns.scatterplot(x='leadtime', y='Recall', 
        #                 hue='region', data=skillscores_df[skillscores_df['region']==region], ax=ax[1])
        # sns.scatterplot(x='leadtime', y='F1', 
        #                 hue='region', data=skillscores_df[skillscores_df['region']==region], ax=ax[2])
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
        # sns.scatterplot(x='leadtime', y='accuracy', hue='region', data=df_predict)
        sns.scatterplot(x='leadtime', y='PoD', 
                        hue='set', data=skillscores_df[skillscores_df['region']==region], ax=ax[0])
        sns.scatterplot(x='leadtime', y='FAR', 
                        hue='set', data=skillscores_df[skillscores_df['region']==region], ax=ax[1])
        # sns.scatterplot(x='leadtime', y='PoFD', 
        #                 hue='set', data=skillscores_df[skillscores_df['region']==region], ax=ax[2])
        # sns.scatterplot(x='leadtime', y='CSI', 
        #                 hue='set', data=skillscores_df[skillscores_df['region']==region], ax=ax[3])
        ax[0].set_title('PoD')
        # ax[0].yaxis.set_visible(False)
        ax[0].grid()
        ax[1].set_title('FAR')
        # ax[1].yaxis.set_visible(False)
        ax[1].grid()
        # ax[2].set_title('PoFD')
        # # ax[2].yaxis.set_visible(False)
        # ax[2].grid()
        # ax[3].set_title('CSI')
        # # ax[3].yaxis.set_visible(False)
        # ax[3].grid()
        plt.setp(ax, ylim=(-0.05,1.05))
        
        # fig.savefig(os.path.abspath(os.path.join("..", "output/maize(enso+rain)/zwe_m2_scores2_%s.pdf"%region)))




#%% PLOT SCORES MAP

for leadtime in leadtimes_val:

    df_predict_dist = skillscores_df[(skillscores_df['region']!='all') & (skillscores_df['leadtime']==leadtime)]
    df_plot = pd.merge(df_predict_dist, zwe2_shp, left_on='region', right_on='ADM1_PCODE', how='right')
    gdf_plot = gpd.GeoDataFrame(df_plot)
    
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,16))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    fig.suptitle('Performance results Model 2 (ENSO+Rain) - Maize areas \n Province level - Lead time %s' %leadtime,
                 fontsize= 22, fontweight= 'bold', x=0.5, y=0.94)   
    divider_ax1 = make_axes_locatable(ax1)
    divider_ax2 = make_axes_locatable(ax2)
    # divider_ax3 = make_axes_locatable(ax3)
    # divider_ax4 = make_axes_locatable(ax4)
    cax1 = divider_ax1.append_axes("right", size="5%", pad=0.2)
    cax2 = divider_ax2.append_axes("right", size="5%", pad=0.2)
    # cax3 = divider_ax3.append_axes("right", size="5%", pad=0.2)
    # cax4 = divider_ax4.append_axes("right", size="5%", pad=0.2)
    
    ax1.set_title('False Alarm Ratio (FAR)', fontsize= 16)
    gdf_plot.plot(ax=ax1, column='FAR', legend= True, 
                  cmap='coolwarm', missing_kwds={'color': "lightgrey",
                                                 "edgecolor": "white",
                                                 "label": "No value"}, 
                  vmin=0, vmax=1, cax=cax1)
    
    ax2.set_title('Proability of Detection (POD)', fontsize= 16)
    gdf_plot.plot(ax=ax2, column='PoD', legend= True, 
                  cmap='coolwarm_r', missing_kwds={'color': "lightgrey",
                                                 "edgecolor": "white",
                                                 "label": "No value"}, 
                  vmin=0, vmax=1, cax=cax2)
    
    # ax3.set_title('Proability of False Detection (POFD)', fontsize= 16)
    # gdf_plot.plot(ax=ax3, column='PoFD', legend= True, 
    #               cmap='coolwarm', missing_kwds={'color': "lightgrey",
    #                                              "edgecolor": "white",
    #                                              "label": "No value"}, 
    #               vmin=0, vmax=1, cax=cax3)
    
    # ax4.set_title('Critical Success Index (CSI)', fontsize= 16)
    # gdf_plot.plot(ax=ax4, column='CSI', legend= True, 
    #               cmap='coolwarm_r', missing_kwds={'color': "lightgrey",
    #                                                "edgecolor": "white",
    #                                                 "label": "No value"}, 
    #               vmin=0, vmax=1, cax=cax4)
    
    
    # fig.savefig(os.path.abspath(os.path.join("..", "output/maize(enso+rain)/zwe_m2_%s_scores2_province.pdf" %leadtime)))
