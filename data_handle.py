# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:10:17 2018

@author: Administrator
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier 
import lightgbm as lgb
import gc
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

TRATIN_PATH = "data/application_train.csv"
TEST_PATH = "data/application_test.csv"

def encode_obiect(data_set):
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0
    # Iterate through the columns
    for col in data_set:
        if data_set[col].dtype == 'object':
            data_set[col] = le.fit_transform(np.array(data_set[col].astype(str)).reshape((-1,)))
            le_count += 1
              
    print('%d columns were label encoded.' % le_count)
    return data_set

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def col_corr():
    #读取数据
    X = pd.read_csv(TRATIN_PATH)
    
    X = encode_obiect(X)
    
    X = X.fillna(X.mean())
    
    names = list(X.columns)
    #train test 数据做对齐
    Y = X['TARGET']
    #app_train['TARGET'] = train_labels
    print ("app_train.dtypes.value_counts():", X.dtypes.value_counts())
    
    
    ## Convert to np arrays
    #app_train = np.array(app_train)
    #app_test = np.array(app_test)
    
    ranks = {}
    
    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)
    ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)
    
    ridge = Ridge(alpha=7)
    ridge.fit(X, Y)
    ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
    
    lasso = Lasso(alpha=.05)
    lasso.fit(X, Y)
    ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
    
    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X, Y)
    ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
    
    rfe = RFE(lr, n_features_to_select=5)
    rfe.fit(X,Y)
    ranks["RFE"] = rank_to_dict(rfe.ranking_, names, order=-1)
    
    rf = RandomForestRegressor()
    rf.fit(X,Y)
    ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
    
    f, pval  = f_regression(X, Y, center=True)
    ranks["Corr."] = rank_to_dict(f, names)
    
    r = {}
    for name in names:
        r[name] = round(np.mean([ranks[method][name] 
                                 for method in ranks.keys()]), 2)
        
    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    methods.append("Mean")
    
    print ("\t%s" % "\t".join(methods))
    for name in names:
        print ("%s\t%s" % (name, "\t".join(map(str, 
                             [ranks[method][name] for method in methods]))))

from sklearn.ensemble import ExtraTreesClassifier 
def ExtraTrees_corr():
    X = pd.read_csv(TRATIN_PATH)
    X = encode_obiect(X)
    X = X.fillna(X.mean())
    Y = X['TARGET']

    clf=ExtraTreesClassifier() 
    clf=clf.fit(X,Y)
    print(clf.feature_importances_)     

def get_data():
    data = pd.read_csv("./data/application_train.csv")
    test = pd.read_csv('./data/application_test.csv')
    prev = pd.read_csv('./data/previous_application.csv')
    buro = pd.read_csv('./data/bureau.csv')
    buro_balance = pd.read_csv('./data/bureau_balance.csv')
    credit_card  = pd.read_csv('./data/credit_card_balance.csv')
    POS_CASH  = pd.read_csv('./data/POS_CASH_balance.csv')
    payments = pd.read_csv('./data/installments_payments.csv')
    
    y = data["TARGET"]
    del data["TARGET"]
    
    categorical_feature = [col for col in data.columns if data[col].dtype == 'object']
    one_hot_df = pd.concat([data, test])
    one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_feature)
    
    data = one_hot_df.iloc[:data.shape[0],:]
    test = one_hot_df.iloc[data.shape[0]:,]
    
    buro_grouped_size = buro_balance.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].size()
    buro_grouped_max = buro_balance.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].max()
    buro_grouped_min = buro_balance.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].min()
    
    buro_counts = buro_balance.groupby("SK_ID_BUREAU")["STATUS"].value_counts(normalize=False)
    buro_counts_unstacked = buro_counts.unstack("STATUS")
    
    buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]
    buro_counts_unstacked["MONTHS_COUNT"] = buro_grouped_size
    buro_counts_unstacked["MONTHS_MIN"] = buro_grouped_min
    buro_counts_unstacked["MONTHS_MAX"] = buro_grouped_max
    
    buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')
    
    prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == "object"]
    prev = pd.get_dummies(prev, columns=prev_cat_features)

    avg_prev = prev.groupby("SK_ID_CURR").mean()
    cnt_prev = prev[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
    avg_prev["nb_app"] = cnt_prev["SK_ID_PREV"]
    del avg_prev["SK_ID_PREV"]
    
    buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
    buro = pd.get_dummies(buro, columns=buro_cat_features)
    avg_buro = buro.groupby('SK_ID_CURR').mean()
    avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
    del avg_buro['SK_ID_BUREAU']
    
    le = LabelEncoder()
    POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
    
    POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
    POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)
    
    le =LabelEncoder()
    credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
    credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
    credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)
    
    avg_payments = payments.groupby("SK_ID_CURR").mean()
    avg_payments2 = payments.groupby("SK_ID_CURR").max()
    avg_payments3 = payments.groupby("SK_ID_CURR").min()
    
    del avg_payments["SK_ID_PREV"]
    
    data = data.merge(right=avg_prev.reset_index(), how="left", on="SK_ID_CURR")
    test = test.merge(right=avg_prev.reset_index(), how="left", on="SK_ID_CURR")
    
    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    data = data.merge(right=POS_CASH.groupby("SK_ID_CURR").mean().reset_index(), how="left", on="SK_ID_CURR")
    test = test.merge(right=POS_CASH.groupby("SK_ID_CURR").mean().reset_index(), how="left", on="SK_ID_CURR")
    data = data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
    data = data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
    data = data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
    data = data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
    
    data["TARGET"] = y
    print("data.shape:", data.shape)    
    data.to_csv("data.csv", index=False)
    test.to_csv("test.csv", index=False)
    return data, y, test

def get_feature(data, test):
    y = data['TARGET']
    data = data.drop(columns = ['TARGET'])
    
    lgbm_params = {
    "boosting":"dart",
    "application":"binary",
    "learning_rate": 0.1,
    'reg_alpha':0.01,
    'reg_lambda': 0.01,
    "n_estimators":10000,
    "max_depth":7,
    "num_leaves":70,
#     "max_bin":550,
    "drop_rate":0.02
    }

    model = lgb.LGBMClassifier(application="binary", boosting_type=lgbm_params["boosting"],
                          learning_rate=lgbm_params["learning_rate"],n_estimators=lgbm_params["n_estimators"],drop_rate=lgbm_params["drop_rate"],
                          num_leaves=lgbm_params["num_leaves"], max_depth=lgbm_params["max_depth"])
    
    feature_importances = np.zeros(data.shape[1])
    for i in range(2): 
        train_data, test_data, train_y, test_y = train_test_split(data, y, test_size=0.2, random_state=i)
        model.fit(train_data, train_y, early_stopping_rounds=100, eval_set=[(test_data, test_y)], eval_metric='auc', verbose=200)
        feature_importances += model.feature_importances_
    
    feature_importances = feature_importances/2
    feature_importances = pd.DataFrame({'feature': list(data.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

    print ("feature_importances head:",feature_importances.head())

    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('There are %d features with 0.0 importance' % len(zero_features))
    print ("feature_importances tail:", feature_importances.tail())
    
    
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index()   
    feature_importances['importance_normalized'] = feature_importances['importance']/feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['importance_normalized'])
        
    norm_feature_importances = feature_importances.copy()
    
    threshold = 0.98
    print ("feature_importances[cumulative_importance]:", norm_feature_importances['cumulative_importance'])
    features_to_keep = list(norm_feature_importances[norm_feature_importances['cumulative_importance'] < threshold]['feature'])
    print ("features_to_keep:", features_to_keep)
    data = data[features_to_keep]
    test = test[features_to_keep]
#    data = data[feature_importances]
#    test = test[feature_importances]
    print('Training shape: ', data.shape)
    print('Testing shape: ', test.shape)
    
    return data, y, test

def valid(data, y, test):
    lgbm_params = {
    "boosting":"dart",
    "application":"binary",
    "learning_rate": 0.01,
    'reg_alpha':1,
    'reg_lambda': 0.1,
    "n_estimators":10000,
    "max_depth":7,
    "num_leaves":70,
#     "max_bin":550,
    "drop_rate":0.02,
    "class_weight":'balanced'
    }

    n_folds = 5
    k_fold = KFold(n_splits=n_folds, shuffle=False, random_state=50)
    
    feature_importances_values = np.zeros(data.shape[1])
    
    test_predictions = np.zeros(test.shape[0])
    out_of_fold = np.zeros(data.shape[0])
    
    valid_scores = []
    train_scores = []
    
    for train_indices, test_indices in k_fold.split(data):
        
        train_data, train_y = data.iloc[train_indices], y.iloc[train_indices]
        test_data, test_y = data.iloc[test_indices], y.iloc[test_indices]
        
        model = lgb.LGBMClassifier(application="binary", boosting_type=lgbm_params["boosting"],
                              learning_rate=lgbm_params["learning_rate"],  n_estimators=lgbm_params["n_estimators"],
                              num_leaves=lgbm_params["num_leaves"],max_depth=lgbm_params["max_depth"],
                              reg_lambda=lgbm_params['reg_lambda'],reg_alpha=lgbm_params["reg_alpha"],
                              drop_rate=lgbm_params["drop_rate"],class_weight=lgbm_params["class_weight"], random_state=50)
        
        model.fit(train_data, train_y, eval_metric='auc', eval_set=[(test_data, test_y), (train_data, train_y)],
                  eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=200)
        
        best_iteration = model.best_iteration_
        feature_importances_values += model.feature_importances_ / k_fold.n_splits
        
        test_predictions += model.predict_proba(test, num_iteration=best_iteration)[:,1]/k_fold.n_splits
        
        out_of_fold[test_indices] = model.predict_proba(test_data, num_iteration = best_iteration)[:, 1]
            
            # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
            
        valid_scores.append(valid_score)
        train_scores.append(train_score)
            
        gc.enable()
        del model, train_data, test_data
        gc.collect()

    submission = pd.DataFrame({'SK_ID_CURR': test["SK_ID_CURR"], 'TARGET': test_predictions})
    submission.to_csv("submissions.csv", index=False)


from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston
def get_feature_by_RandomizedLasso(data, test):
    data = data.fillna(data.mean())
    
    Y = data['TARGET']
    X = data.drop(columns = ['TARGET'])
    
    names = data.drop(columns = ['TARGET'])
    
    print("X.shape:", X.shape)
    print("test.shape:", test.shape)
    
    X = pd.get_dummies(X)
    
    rlasso = RandomizedLasso(alpha=0.025)
    rlasso.fit(X, Y)
     
    print ("Features sorted by their score:")
    print (sorted(zip(map(lambda x: round(x, 3), rlasso.scores_), 
                     names), reverse=True))

if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    test = pd.read_csv('test.csv')
    data, y, test = get_feature(data, test)
    valid(data, y, test)

    
#    get_feature_by_RandomizedLasso(data, test)
    
    
    