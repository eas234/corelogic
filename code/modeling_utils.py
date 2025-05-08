import logging
import miceforest as mf
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import TrialState

def subset_cols(X, n_non_null=10):
	
    """
    Subset inputs to features that meet the following conditions:
    1. Contain more than 1 unique value (not including nans)
    2. Contain at least n non-null values
    """
	
    drop_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    if drop_cols:
        print('Warning: ' + str(len(drop_cols)) + ' columns have only one value. Dropping these columns:')
        print(drop_cols)
        X = X.drop(columns=drop_cols) 

    mostly_null_cols = [col for col in X.columns if X[col].notnull().sum() <= n_non_null]
    if mostly_null_cols:
        print('Warning: ' + str(len(mostly_null_cols)) + ' columns have fewer than ' + str(n_non_null) + ' non-null values. Dropping these columns:')
        print(mostly_null_cols)
        X = X.drop(columns=mostly_null_cols)

    X = X.reset_index(drop=True)

    return X

def impute_and_normalize(X, 
                         features_to_preprocess, 
                         random_state=42,
                         mice_iters=3):
    """
    Imputes and normalizes selected features in a DataFrame using miceforest and StandardScaler.

    Parameters:
    - X: pd.DataFrame — input features.
    - features_to_preprocess: list of column names to impute and normalize.
    - random_state: int — for reproducibility.

    Returns:
    - pd.DataFrame with imputed and normalized features.
    """

    # Copy the DataFrame to avoid modifying original
    X_processed = X.copy()

    # Subset the data to be imputed
    features_to_preprocess = [col for col in features_to_preprocess if col in X_processed.columns]
    subset = X_processed[features_to_preprocess]

    # Create miceforest kernel
    kernel = mf.ImputationKernel(
        subset,
        num_datasets=1,
        save_all_iterations_data=False,
        random_state=random_state
    )

    # Perform imputation
    kernel.mice(mice_iters)

    # Extract the imputed dataset
    imputed_subset = kernel.complete_data(0)

    # Normalize using StandardScaler
    scaler = StandardScaler()
    normalized_array = scaler.fit_transform(imputed_subset)

    # Replace in original DataFrame
    X_processed[features_to_preprocess] = normalized_array

    return X_processed

def get_data_splitter(X, y, meta, test_size=0.2, random_state=42):
    # Split happens here ONCE
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(X, y, meta, test_size=test_size, random_state=random_state)
    
    # Closure: inner function remembers variables from the outer scope
    def get_split():
        return X_train, X_test, y_train, y_test, meta_train, meta_test

    return get_split

def mpe2_loss(y_true, 
              y_pred):
    '''
    Mean Percentage Error Squared loss function
    for use as alternative to MSE in model dev.
    '''

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    loss = (((y_true - y_pred)**2 / (y_true)**2).sum())/len(y_true)
                  
    return loss

def mse_loss(y_true, y_pred):
    '''
    Mean squared error loss function for use in model dev
    '''
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    loss = ((y_true - y_pred)**2)/len(y_true)

    return loss

def rf_reg_objective(trial, 
              X_train, 
              y_train,
              test_size=0.2, 
              random_state=42, 
              loss_func=mpe2_loss, 
              n_jobs=4,
              cv_folds=5):

    '''
    Random forest regressor objective for use in optuna 
    bayesian hyperparameter tuning pipeline
    '''
                  
    # Number of trees in random forest
    n_estimators = trial.suggest_int(name="n_estimators", low=100, high=500, step=100)

    # Number of features to consider at every split
    max_features = trial.suggest_categorical(name="max_features", choices=['sqrt', 'log2', None]) 

    # Maximum number of levels in tree
    max_depth = trial.suggest_int(name="max_depth", low=10, high=110, step=20)

    # Minimum number of samples required to split a node
    min_samples_split = trial.suggest_int(name="min_samples_split", low=2, high=10, step=2)

    # Minimum number of samples required at each leaf node
    min_samples_leaf = trial.suggest_int(name="min_samples_leaf", low=1, high=4, step=1)
    
    params = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf
    }

    # build model and scorer            
    model = RandomForestRegressor(random_state=random_state, **params)
    scorer = make_scorer(loss_func)
    cv_score = cross_val_score(model, X_train, y_train, scoring=scorer, n_jobs=n_jobs, cv=cv_folds)
    mean_cv_accuracy = cv_score.mean()
                  
    return mean_cv_accuracy

def tune_model(X_train, 
	    y_train, 
	    study_name="example-study",
	    load_if_exists=True,
	    sampler=TPESampler(seed=42),
	    sampler_path='sampler.pkl', #.pkl file
	    params_path='best_params.pkl', #.pkl file
	    trials_path='trials.csv', #.csv file
	    n_trials=20,
	    test_size=0.2,
	    random_state=42,
	    loss_func=mpe2_loss,
	    n_jobs=4,
	    cv_folds=5):

    '''
    Model tuner
    '''
    # define storage name
    storage_name="sqlite:///{}.db".format(study_name)
              
    # check if sampler is saved from a previous run and load if it is
    if os.path.isfile(sampler_path):
        with open(sampler_path, "rb") as f:
            restored_sampler = pickle.load(f)
        
            study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=load_if_exists,
            sampler=restored_sampler
        )
        completed_trials = [x for x in study.trials if x.state == TrialState.COMPLETE]
        prev_trials = len(completed_trials) 
    
    else:
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage_name, 
            load_if_exists=load_if_exists, 
            sampler=sampler
        )
        prev_trials = 0

    # Run optimization
    y_train = np.ravel(y_train)
    if prev_trials < n_trials:
        study.optimize(lambda trial: rf_reg_objective(trial, X_train, y_train, test_size=test_size, random_state=random_state,loss_func=loss_func, n_jobs=n_jobs, cv_folds=cv_folds), n_trials=(n_trials-prev_trials))

    # Save the sampler
    with open(sampler_path, "wb") as fout:
        pickle.dump(study.sampler, fout)

    # Save the best parameters
    with open(params_path, 'wb') as fout:
        pickle.dump(study.best_params, fout)

    # Output trials DataFrame
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(trials_path)
                
    return df

def rf_train_test_write(X_train, X_test, y_train, y_test, meta_train, meta_test, params_path='params_path.pkl', out_path='out_path.csv'):

    """
    Train model using optimal hyperparams
    Write out predictions along with ground truth and metadata to specified directory
    """

    with open(params_path, 'rb') as f:
         hyperparams=pickle.load(f)

    model = RandomForestRegressor(**hyperparams)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # align indices
    meta_test = meta_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    results_df = pd.DataFrame(meta_test)
    results_df['y_true'] = y_test
    results_df['y_pred'] = y_pred

    results_df.to_csv(out_path, index=False)

    return results_df
