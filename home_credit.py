# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:09:40 2018

@author: Administrator
"""

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest

TRATIN_PATH = "data/application_train.csv"
TEST_PATH = "data/application_test.csv"


def encode_obiect(data_set):
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0
    # Iterate through the columns
    for col in data_set:
        if data_set[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(data_set[col].unique())) <= 2:
                # Train on the training data
                le.fit(data_set[col])
                # Transform both training and testing data
                data_set[col] = le.transform(data_set[col])          
                # Keep track of how many columns were label encoded
                le_count += 1
              
    print('%d columns were label encoded.' % le_count)
    return data_set

def missing_values(df):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(df)
    return imp.transform(df)

def feat_ext_source(df):
    medi_avg_mode = [f_ for f_ in df.columns if '_AVG' in f_ or '_MODE' in f_ or '_MEDI' in f_]
    df.drop(medi_avg_mode, axis=1, inplace=True)
    return df

    
def get_data():
    app_train = pd.read_csv('data/application_train.csv')
    app_test = pd.read_csv('data/application_test.csv')
    
    app_train = encode_obiect(app_train)
    app_test = encode_obiect(app_test)
    
    # one-hot encoding of categorical variables
    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)
    
    train_labels = app_train['TARGET']
    # Align the training and testing data, keep only columns present in both dataframes
    app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
    app_train['TARGET'] = train_labels

    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    
    app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
    app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

#    app_train = feat_ext_source(app_train)
#    app_test = feat_ext_source(app_test)

    # Find correlations with the target and sort
    print("before train shape:", app_train.shape)
    correlations = app_train.corr()['TARGET']
#    print("before correlations:", correlations[correlations.abs() > 0.001])
    feature_importances = correlations[correlations.abs() < 0.001]
    print("feature_importances:", feature_importances)
    app_train = app_train.drop(feature_importances.index, axis = 1)
    print("after train shape:", app_train.shape)
#    correlations.drop(correlations.abs().values < 0.001)
#    print("before correlations:", correlations.abs())
    
    # Display correlations
#    print('Most Positive Correlations:\n', correlations.tail(15))
#    print('\nMost Negative Correlations:\n', correlations.head(15))
    
    return app_train, app_test


def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
#    features = missing_values(features)
#    test_features = missing_values(test_features)
    
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    print (" k_fold.n_splits:",  k_fold.n_splits)
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=15000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.001, 
                                   reg_alpha = 0.1, reg_lambda = 0.01, 
                                   subsample = 0.8, n_jobs = -1, random_state = 0)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        print ("best_iteration:", best_iteration)
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        print ("model.predict_proba(test_features, num_iteration = best_iteration)[:, 1]:",
               model.predict_proba(test_features, num_iteration = best_iteration)[:, 1])
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        print (" model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]:",
                model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1])
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics


def handle_data(app_train, app_test):
    poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
    poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    
    imputer = Imputer(strategy = 'median')
    poly_target = poly_features['TARGET']
    poly_features = poly_features.drop(columns = ['TARGET'])
    
    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)
    
    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree = 3)
    # Train the polynomial features
    poly_transformer.fit(poly_features)
    poly_transformer.fit(poly_features_test)
    
    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)

    poly_features = pd.DataFrame(poly_features, 
                                 columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))
    # Add in the target
    poly_features['TARGET'] = poly_target
    
    poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))
    
    # Merge polynomial features into training dataframe
    poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
    app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')
    
    # Merge polnomial features into testing dataframe
    poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
    app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')
    
    # Align the dataframes
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)
    
    app_train_poly = pd.concat([app_train_poly, poly_target], axis=1)
    
    print ("app_train_poly shape:", app_train_poly.shape)
    print ("app_test_poly shape:", app_test_poly.shape)
    
    return app_train_poly, app_test_poly
    
    
def save_as_csv(data):
    data.to_csv('lgb.csv', index = False)


def StackingClassifier(features, test_features, encoding = 'ohe', n_folds = 5):
    features = features.fillna(features.mean())
    test_features = test_features.fillna(test_features.mean())
    
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)      
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)       
        # No categorical indices to record
        cat_indices = 'auto'  
    # Integer label encoding
    elif encoding == 'le':        
        # Create a label encoder
        label_encoder = LabelEncoder()     
        # List for storing categorical indices
        cat_indices = []
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)  
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    clfs = [lgb.LGBMClassifier(n_estimators=100, objective = 'binary', 
                               class_weight = 'balanced', learning_rate = 0.01, 
                               reg_alpha = 0.1, reg_lambda = 0.1, 
                               subsample = 0.8, n_jobs = -1, random_state = 50),
            RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1, criterion='gini'),
    #        RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
        ]

    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
#    features = missing_values(features)
#    test_features = missing_values(test_features)
      
    skf = list(StratifiedKFold(labels, n_folds))
    # Empty array for feature importances
    feature_importance_values = np.zeros((features.shape[0], len(clfs)))
    
    # Empty array for test predictions
    test_predictions = np.zeros((test_features.shape[0], len(clfs)))
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    print ("Creating train and test sets for blending.")    
    
    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((test_features.shape[0], len(skf)))
        for i,(train_indices, valid_indices) in enumerate(skf):
            print ()
            # Training data for the fold
            train_features, train_labels = features[train_indices], labels[train_indices]
            # Validation data for the fold
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]
            clf.fit(train_features, train_labels)
            y_submission = clf.predict_proba(valid_features)[:, 1]
            feature_importance_values[valid_indices, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(test_features)[:, 1]
        test_predictions[:, j] = dataset_blend_test_j.mean(1)

    clf = LogisticRegression()
    clf.fit(feature_importance_values, labels)
    y_submission = clf.predict_proba(test_predictions)[:, 1]
    
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    
    print ("y_submission:", y_submission)
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': y_submission})
    
    return submission

if __name__ == '__main__':
     data_train, data_test = get_data()
     
#     submission = StackingClassifier(data_train, data_test, encoding = 'ohe', n_folds = 5)   
#     save_as_csv(submission)   
     
#     data_train, data_test = handle_data(data_train, data_test)
#     
#     print ("data_train:", data_train.columns)
#     print ("data test:", data_test.columns)
#     
#     submission, fi, metrics = model(data_train, data_test, encoding = 'le')
#     print('Baseline metrics')
#     print(metrics)
#     save_as_csv(submission)
     