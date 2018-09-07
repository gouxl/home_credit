# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:52:17 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
#from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from hpsklearn import HyperoptEstimator
from hyperopt import tpe
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
from gcforest.gcforest import GCForest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('./data/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('./data/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('./data/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('./data/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('./data/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('./data/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('./data/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('./data/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

#OMP_NUM_THREADS=1
#export OMP_NUM_THREADS

def find_best_param(train_x, train_y):
    lgb = LGBMClassifier()
    with timer("find best param"):
        clf = HyperoptEstimator(classifier=lgb, algo=tpe.suggest, max_evals=100, seed=0)
        print("X_test shape:", train_x.shape)
        print("y_test shape:", train_y.shape)
        clf.fit(train_x, train_y)
        print ("new score", clf.score(train_x, train_y))
        print ('new model:', clf.best_model())
        
def hyperopt_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
        
#    correlations = train_df.corr()['TARGET']
#    feature_importances = correlations[correlations.abs() < 0.001]
#    train_df = train_df.drop(feature_importances.drop('SK_ID_CURR').index, axis = 1)
#    test_df = test_df.drop(feature_importances.drop('SK_ID_CURR').index, axis = 1)
#    print("corr. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
       
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        print("train_x:", train_x.head())
        print("train_y:", train_y)
        # LightGBM parameters found by Bayesian optimization
        lgb = LGBMClassifier()
        clf = HyperoptEstimator(classifier=lgb, algo=tpe.suggest, max_evals=100)
        clf.fit(train_x, train_y)
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
#    display_importances(feature_importance_df)
    return feature_importance_df
# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
#    train_df = df[df['TARGET'].notnull()]
#    test_df = df[df['TARGET'].isnull()]
#    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
#    del df
#    gc.collect()   
    
    train_df = pd.read_csv('train_df.csv')
    test_df = pd.read_csv('test_df.csv')
    print("train_df shape A:", train_df.shape)
    feature_names = list(train_df.columns)
    
    #人工删除 空值大于50%的特征
    drop_feature_names = []
    for i in feature_names:
        a = train_df[i].isna().value_counts()
        if a[False] < train_df.shape[0]/2:
 #           train_df.drop(columns = [i], inplace = True)
            drop_feature_names.append(i)
    
    train_df.drop(columns = drop_feature_names, inplace = True)
    test_df.drop(columns = drop_feature_names, inplace = True)
    print("train_df shape B:", train_df.shape)
    print("test_df shape B:", test_df.shape)
    
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
        
    train_df, test_df = get_feature(train_df, test_df)
    print("train shape:", train_df.shape)
    print("test shape:", test_df.shape)
    
    labels = train_df['TARGET']
    
    train_ids = train_df['SK_ID_CURR']
    test_ids = test_df['SK_ID_CURR']
    
    train_columns = train_df.columns
    test_columns = test_df.columns
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    train_df = np.array(train_df)
    test_df = np.array(test_df)
    labels = np.array(labels)
    
    train_df = np.nan_to_num(train_df)
    test_df = np.nan_to_num(test_df)
    
    train_df =  min_max_scaler.fit_transform(train_df)
    test_df =  min_max_scaler.fit_transform(test_df)
    
    train_df = pd.DataFrame(train_df, index = train_ids, columns = train_columns)
    test_df = pd.DataFrame(test_df, index = test_ids, columns = test_columns)
#    correlations = train_df.corr()['TARGET']
#    print("correlations:", correlations.abs())
#    feature_importances = correlations[correlations.abs() < 0.01]
#    print("feature_importances:", feature_importances)
#    
#    print ("correlations count():", correlations.count())
#    print ("feature_importances count():", feature_importances.count())
#    
#    train_df = train_df.drop(feature_importances.index, axis = 1)
#    test_df = test_df.drop(feature_importances.index, axis = 1)
#    print("corr. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            #scale_pos_weight=11
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': sub_preds})
        submission.to_csv("submission.csv", index= False)
#        test_df['TARGET'] = sub_preds
#        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
#    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

def get_feature(train, test):    
    
#    data = df[df['TARGET'].notnull()]
#    test = df[df['TARGET'].isnull()]

    print("data shape:", train.shape)
    print("test shape:", test.shape)
    
    y = train['TARGET']
    train = train.drop(columns = ['TARGET'])
    
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']
    
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

    model = LGBMClassifier(application="binary", boosting_type=lgbm_params["boosting"],
                          learning_rate=lgbm_params["learning_rate"],n_estimators=lgbm_params["n_estimators"],drop_rate=lgbm_params["drop_rate"],
                          num_leaves=lgbm_params["num_leaves"], max_depth=lgbm_params["max_depth"])
    
    feature_importances = np.zeros(train.shape[1])
    for i in range(2): 
        train_data, test_data, train_y, test_y = train_test_split(train, y, test_size=0.2, random_state=i)
        model.fit(train_data, train_y, early_stopping_rounds=100, eval_set=[(test_data, test_y)], eval_metric='auc', verbose=200)
        feature_importances += model.feature_importances_
    
    feature_importances = feature_importances/2
    feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

    print ("feature_importances head:",feature_importances.head())

    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('There are %d features with 0.0 importance' % len(zero_features))
    print ("feature_importances tail:", feature_importances.tail())
    
    
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index()   
    feature_importances['importance_normalized'] = feature_importances['importance']/feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['importance_normalized'])
        
    norm_feature_importances = feature_importances.copy()
    
    threshold = 0.99
    print ("feature_importances[cumulative_importance]:", norm_feature_importances['cumulative_importance'])
    features_to_keep = list(norm_feature_importances[norm_feature_importances['cumulative_importance'] < threshold]['feature'])
    print ("features_to_keep:", features_to_keep)
    train = train[features_to_keep]
    test = test[features_to_keep]
#    data = data[feature_importances]
#    test = test[feature_importances]
    print('Training shape: ', train.shape)
    print('Testing shape: ', test.shape)
    
    train['TARGET'] = y
    train['SK_ID_CURR'] = train_ids
    test['SK_ID_CURR'] = test_ids
    return train, test

def choose_lgb_model(df):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
#    del df
    gc.collect()
    
    y_train = train_df['TARGET']
    X_train = train_df.drop(columns = ['TARGET'])
    

    tuned_params = [{'objective': ['binary'], 
                     'n_estimators': range(5000,15000,3000),
                     'colsample_bytree':np.linspace(0.5,0.98,10),
                     'subsample':np.linspace(0.7,0.9,4),
                     'max_depth':range(5,15,2),
                     'num_leaves':range(10,40,5)}]

    estimator  = LGBMClassifier(
            nthread=4,
            #is_unbalance=True,
            learning_rate=0.02,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            #scale_pos_weight=11
            )
        
    clf = GridSearchCV(estimator, tuned_params, scoring='roc_auc', cv=5)
    clf.fit(X_train, y_train)
    
    print ('current best parameters of lgb: ',clf.best_params_)
    return clf.best_estimator_


def choose_xgb_model(df):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting xgb. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
#    del df
    gc.collect()
    
    y_train = train_df['TARGET']
    X_train = train_df.drop(columns = ['TARGET'])
    
    tuned_params = [{'objective': ['binary:logistic'], 
                     'learning_rate': [0.01,0.1,0.5], 
                     'subsample':np.linspace(0.7,0.9,4),
                     'n_estimators': range(5000,15000,30000),
                     'colsample_bytree':np.linspace(0.5,0.98,10),
                     'min_child_weight':range(10,50,10),
                     'max_depth':range(5,15,5)}]
    
    estimator  = xgb.XGBClassifier(nthread=4,n_jobs=-1)
        
    clf = GridSearchCV(estimator, tuned_params, scoring='roc_auc',cv=5)
    clf.fit(X_train, y_train)
    
    print ('current best parameters of xgboost: ',clf.best_params_)
    return clf.best_estimator_

def VotingClassifier_model(df, bst_lgb, bst_xgb, bst_cat):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting VotingClassifier. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    
    # Cross validation mode 
    folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=47)
     
    correlations = train_df.corr()['TARGET']
    feature_importances = correlations[correlations.abs() < 0.001]
    print("feature_importances index:", feature_importances.index)
    train_df = train_df.drop(feature_importances.index, axis = 1)
    test_df = test_df.drop(feature_importances.index, axis = 1)
    print("corr. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    # Create arrays and dataframes to store results 
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        
        clf = VotingClassifier(estimators=[('xgb', bst_xgb), ('lgb', bst_lgb), ('cat', bst_cat)], 
                                            voting='soft', weights=[1, 2, 1])
        clf.fit(train_x, train_y)
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
#    display_importances(feature_importance_df)

def missing_values(df):
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(df)
    return imp.transform(df)


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
#    ca_config["estimators"].append(
#            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
#             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1, "num_class": 2} )
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier",        
            "objective": "binary:logistic",
            "learning_rate": 0.01,
            "booster":"gbtree",
            "eval_metric":"auc",
            "eta": 0.025,
            "max_depth": 6,
            "min_child_weight":19,
            "gamma" : 0,
            "subsample": 0.8,
            "colsample_bytree": 0.632,
            "reg_alpha":0,
            "reg_lambda":0.05,
            "nrounds":2000}
            )
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "RandomForestClassifier",
             "bootstrap":"True",
             "criterion":"gini",
             "min_samples_split":2,
             "min_samples_leaf":1,
             "max_features": "auto",
             "n_estimators": 10,
             "max_depth": None,
             "n_jobs": -1})
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "ExtraTreesClassifier",
             "n_estimators": 10, 
             "max_depth": None,
             "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config

def gcforest_modle():
#    train_df = df[df['TARGET'].notnull()]
#    test_df = df[df['TARGET'].isnull()]
#    del df
    
#    train_df.to_csv("train_df.csv", index= False)
#    test_df.to_csv("test_df.csv", index= False)
    
    train_df = pd.read_csv('train_df.csv')
    test_df = pd.read_csv('test_df.csv')
    print("train_df shape A:", train_df.shape)
    feature_names = list(train_df.columns)
    
    #人工删除 空值大于50%的特征
    drop_feature_names = []
    for i in feature_names:
        a = train_df[i].isna().value_counts()
        if a[False] < train_df.shape[0]/2:
 #           train_df.drop(columns = [i], inplace = True)
            drop_feature_names.append(i)
    print("drop_feature_names:", drop_feature_names)
    
    train_df.drop(columns = drop_feature_names, inplace = True)
    test_df.drop(columns = drop_feature_names, inplace = True)
    print("train_df shape B:", train_df.shape)
    print("test_df shape B:", test_df.shape)
   
    train_df, test_df = get_feature(train_df, test_df)
    print("data shape:", train_df.shape)
    print("test shape:", test_df.shape)

    
    labels = train_df['TARGET']
    
    train_ids = train_df['SK_ID_CURR']
    test_ids = test_df['SK_ID_CURR']
    
    train_df = train_df.drop(columns = ['TARGET','SK_ID_CURR','index'])
    test_df = test_df.drop(columns = ['SK_ID_CURR','index'])
    

    min_max_scaler = preprocessing.MinMaxScaler()
    
    train_df = np.array(train_df)
    test_df = np.array(test_df)
    labels = np.array(labels)
    
    train_df = np.nan_to_num(train_df)
    test_df = np.nan_to_num(test_df)

    train_df =  min_max_scaler.fit_transform(train_df)
    test_df =  min_max_scaler.fit_transform(test_df)
    

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    
    X_train, X_test, y_train, y_test = train_test_split(train_df, labels, test_size=0.2, random_state=0)
    
    config = get_toy_config()
    gc = GCForest(config)
#    gc.set_keep_model_in_mem(False)
    
    gc.fit_transform(train_df, labels)
    y_pred = gc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
           
    sub_preds += gc.predict_proba(test_df)[:, 1]
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': sub_preds})
    submission.to_csv("gc_forest.csv", index= False)
    
def blend_modle():
    train_df = pd.read_csv('train_df.csv')
    test_df = pd.read_csv('test_df.csv')
    print("train_df shape A:", train_df.shape)
    feature_names = list(train_df.columns)
    
    #人工删除 空值大于50%的特征
    drop_feature_names = []
    for i in feature_names:
        a = train_df[i].isna().value_counts()
        if a[False] < train_df.shape[0]/2:
     #           train_df.drop(columns = [i], inplace = True)
            drop_feature_names.append(i)
    print("drop_feature_names:", drop_feature_names)
    
    train_df.drop(columns = drop_feature_names, inplace = True)
    test_df.drop(columns = drop_feature_names, inplace = True)
    print("train_df shape B:", train_df.shape)
    print("test_df shape B:", test_df.shape)
       
    labels = train_df['TARGET']
    
    train_ids = train_df['SK_ID_CURR']
    test_ids = test_df['SK_ID_CURR']
    
    train_df = train_df.drop(columns = ['TARGET','SK_ID_CURR','index'])
    test_df = test_df.drop(columns = ['TARGET','SK_ID_CURR','index'])
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    train_df = np.array(train_df, dtype=np.float)
    test_df = np.array(test_df, dtype=np.float)
    labels = np.array(labels, dtype=np.float)
    
    print ("labels:",labels)
    train_df = np.nan_to_num(train_df)
    test_df = np.nan_to_num(test_df)
    
    train_df =  min_max_scaler.fit_transform(train_df)
    test_df =  min_max_scaler.fit_transform(test_df)
    
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            CatBoostClassifier(bootstrap_type='Bernoulli', depth=7),
            xgb.XGBClassifier(objective = "binary:logistic",booster = "gbtree", max_depth = 6),
            LGBMClassifier(n_estimators=100, n_jobs=-1, learning_rate=0.02),           
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
    n_folds = 10
    skf = list(StratifiedKFold(labels, n_folds))
    
    dataset_blend_train = np.zeros((train_df.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test_df.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((test_df.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print ("Fold", i)
            X_train = train_df[train]
            y_train = labels[train]
            X_test = train_df[test]
            y_test = labels[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(test_df)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    
    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, labels)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
    
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': y_submission})
    submission.to_csv("blend_1.csv", index= False)
    
    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': y_submission})
    submission.to_csv("blend_2.csv", index= False)
    
    
    
    
def main(debug = False):
#    num_rows = 10000 if debug else None
#    df = application_train_test(num_rows)
#    with timer("Process bureau and bureau_balance"):
#        bureau = bureau_and_balance(num_rows)
#        print("Bureau df shape:", bureau.shape)
#        df = df.join(bureau, how='left', on='SK_ID_CURR')
#        del bureau
#        gc.collect()
#    with timer("Process previous_applications"):
#        prev = previous_applications(num_rows)
#        print("Previous applications df shape:", prev.shape)
#        df = df.join(prev, how='left', on='SK_ID_CURR')
#        del prev
#        gc.collect()
#    with timer("Process POS-CASH balance"):
#        pos = pos_cash(num_rows)
#        print("Pos-cash balance df shape:", pos.shape)
#        df = df.join(pos, how='left', on='SK_ID_CURR')
#        del pos
#        gc.collect()
#    with timer("Process installments payments"):
#        ins = installments_payments(num_rows)
#        print("Installments payments df shape:", ins.shape)
#        df = df.join(ins, how='left', on='SK_ID_CURR')
#        del ins
#        gc.collect()
#    with timer("Process credit card balance"):
#        cc = credit_card_balance(num_rows)
#        print("Credit card balance df shape:", cc.shape)
#        df = df.join(cc, how='left', on='SK_ID_CURR')
#        del cc
#        gc.collect()
#    with timer("Process gc forest"):
#        gcforest_modle()
#    with timer("lgb modle"):
#        bst_lgb = choose_lgb_model(df)
#        gc.collect()
#    with timer("xgb modle"):
#        bst_xgb = choose_xgb_model(df)
#        gc.collect()
#    with timer("VotingClassifier"):
#        bst_lgb = LGBMClassifier(
#            nthread=4,
#            #is_unbalance=True,
#            n_estimators=10000,
#            learning_rate=0.02,
#            num_leaves=32,
#            colsample_bytree=0.9497036,
#            subsample=0.8715623,
#            max_depth=8,
#            reg_alpha=0.04,
#            reg_lambda=0.073,
#            min_split_gain=0.0222415,
#            min_child_weight=40,
#            silent=-1,
#            verbose=-1,
#            #scale_pos_weight=11
#            )
#        bst_xgb = xgb.XGBClassifier(
#            objective = "binary:logistic",
#            booster = "gbtree",
#            eval_metric = "auc",
#            nthread = 4,
#            eta = 0.025,
#            max_depth = 6,
#            min_child_weight = 19,
#            gamma = 0,
#            subsample = 0.8,
#            colsample_bytree = 0.632,
#            reg_alpha = 0,
#            reg_lambda = 0.05,
#            nrounds = 2000
#          )
#        bst_cat = CatBoostClassifier(
#            iterations=1000,
#            learning_rate=0.01,
#            depth=7,
#            l2_leaf_reg=40,
#            bootstrap_type='Bernoulli',
#            subsample=0.7,
#            scale_pos_weight=5,
#            eval_metric='AUC',
#            metric_period=50,
#            od_type='Iter',
#            od_wait=45,
#            allow_writing_files=False)
#        
#        VotingClassifier_model(df, bst_lgb, bst_xgb, bst_cat)
#    with timer("hyperopt lightgbm"):
#        feat_importance = hyperopt_lightgbm(df, num_folds= 5, stratified= True, debug= debug)
        
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(num_folds= 5, stratified= True, debug= debug)
#    with timer("run blending"):
#        blend_modle()

if __name__ == "__main__":    
    submission_file_name = "submission.csv"
    with timer("Full model run"):
        main()