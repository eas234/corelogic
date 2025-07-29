import joblib
import logging
import math
import miceforest as mf
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
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

import lightgbm as lgb

def create_directories_if_missing(directories):
    """
    Takes a list of directory paths and ensures each exists.
    If a directory doesn't exist, it will be created.

    Parameters:
    directories (list of str): List of directory paths.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def mpe2_loss(y_true, 
              y_pred):
    '''
    Mean Percentage Error Squared loss function
    for use as alternative to MSE in model dev.

    NOTE: generally should not use this as hessian/gradient are 
    unstable in lightGBM implementation.

    inputs:
    - y_true: sale price
    - y_pred: assessed value

    outputs:
    - loss: mean squared percentage error
    '''

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    loss = (((y_true - y_pred)**2 / (y_true)**2).sum())/len(y_true)
                  
    return loss

def mse_loss(y_true, y_pred):
    '''
    Mean squared error loss function for use in model dev

    inputs:
    - y_true: sale price
    - y_pred: assessed value

    outputs:
    - loss: mean squared error
    
    '''
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    loss = (((y_true - y_pred)**2).sum())/len(y_true)

    return loss

def mae_loss(y_true, y_pred):
    """
    Mean absolute error loss function for use in modeling pipelines

    inputs:
    - y_true: sale price
    - y_pred: assessed value

    outputs:
    - loss: mean absolute error
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    abs_err = np.ravel([abs(x-y) for x, y in zip(y_true, y_pred)])
    loss = abs_err.mean()

    return loss

def rf_reg_objective(trial, 
              X_train, 
              y_train,
              random_state=42, 
              loss_func=mse_loss, 
              n_jobs=4,
              cv_folds=5):

    '''
    Random forest regressor objective for use in optuna 
    bayesian hyperparameter tuning pipeline

    inputs:
    - trial: optuna trial instance
    - X_train: training set features
    - y_train: training set labels
    - random_state: for reproducibility across runs
    - loss_func: desired loss function. can be mean squared percentage error (mpe2) or mean squared error (mse)
    - n_jobs: number of cpus to use in parallel while tuning
    - cv_folds: how many folds you want in the test set for cross-validation

    outputs:
    - mean_cv_accuracy: average performance of trial's parameters across cv_folds
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

def lasso_objective(trial,
		   X_train,
		   y_train,
		   random_state=42,
		   loss_func=mse_loss,
		   n_jobs=4,
		   cv_folds=5):
    """
    Lasso objective function for use in optuna pipeline.

    inputs:
    - trial: optuna trial instance
    - X_train: training set features
    - y_train: training set labels
    - random_state: for reproducibility across runs
    - loss_func: desired loss function. can be mean squared percentage error (mpe2) or mean squared error (mse)
    - n_jobs: number of cpus to use in parallel while tuning
    - cv_folds: how many folds you want in the test set for cross-validation

    outputs:
    - mean_cv_accuracy: average performance of trial's parameters across cv_folds
    """

    # alpha/lambda weight on lasso regularizer term
    alpha = trial.suggest_float("alpha", 1e-6, 1000, log=True)

    # determines how the model updates coefficients at every iter
    selection = trial.suggest_categorical(name="selection", choices=['random', 'cyclic']) 

    # 
    max_iter = trial.suggest_int("max_iter", 1000, 10000, step=1000)

    params = {
        "alpha": alpha,
        "selection": selection,
        "max_iter": max_iter
    }

    # build model and scorer
    model = linear_model.lasso(random_state=random_state, **params)
    scorer = make_scorer(loss_func)
    cv_score = cross_val_score(model, X_train, y_train, scoring=scorer, n_jobs=n_jobs, cv=cv_folds)

    mean_cv_accuracy = cv_score.mean()
                  
    return mean_cv_accuracy

def lightGBM_objective(trial, 
              X_train, 
              y_train,
              random_state=42, 
              cv_folds=5):

    '''
    lightGBM objective for use in optuna 
    bayesian hyperparameter tuning pipeline

    inputs:
    - trial: optuna trial instance
    - X_train: training set features
    - y_train: training set labels
    - random_state: for reproducibility across runs
    - cv_folds: how many folds you want in the test set for cross-validation

    outputs:
    - min(cv_results['l1-mean']): average performance of trial's parameters across cv_folds.
    l1 is MAE loss, which is the only loss function accepted by this model for now.
    '''
    ## hyperparam space 
    # reducing num_iterations relative to ccao to speed up training time
    num_iterations = trial.suggest_int(name='num_iterations', low=100, high=700, step=200)

    learning_rate = trial.suggest_float(name='learning_rate', low=0.001, high=0.398, log=True)

    max_bin = trial.suggest_int(name='max_bin', low=100, high=562, step=66)

    num_leaves = trial.suggest_int('num_leaves', 31, 255, step=32)

    max_depth = trial.suggest_int('max_depth', 4, 12)

    # high relative to ccao due to lower-dimensional feature space
    feature_fraction = trial.suggest_float(name='feature_fraction', low=0.7, high=1.0, step=0.1)

    # when min_gain_to_split was too high, models with very few features never split. reduced here relative to ccao
    min_gain_to_split = trial.suggest_float('min_gain_to_split', 0.0, 0.03, log=False)

    # reduced relative to ccao because models with too high min_data_in_leaf never split
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 35)
    
    min_sum_hessian_in_leaf = trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 0.1, log=True)

    lambda_l1 = trial.suggest_float(name='lambda_l1', low=0.001, high=100, log=True)

    lambda_l2 = trial.suggest_float(name='lambda_l2', low=0.001, high=100, log=True)
      
    params = {'objective': "regression",
        'metric': 'mae', # only loss function accepted for now
        'num_iterations': num_iterations,
        'learning_rate': learning_rate,
        'max_bin': max_bin,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'feature_fraction': feature_fraction,
        'min_gain_to_split': min_gain_to_split,
        'min_data_in_leaf': min_data_in_leaf,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'early_stopping_rounds': 20 # set relatively low to help limit compute time
    }
    
    # build model and scorer      

    dtrain = lgb.Dataset(X_train, label=y_train)
    cv_results = lgb.cv(
        params,
        dtrain,
        metrics='mae',
        nfold=cv_folds,
        stratified=False,
        shuffle=False,
        seed=random_state
    )
              
    return min(cv_results['valid l1-mean'])

def tune_model(X_train, 
	    y_train, 
	    study_name="example-study",
	    load_if_exists=True,
	    sampler=TPESampler(seed=42, n_startup_trials=20, multivariate=True),
	    sampler_path='sampler.pkl', #.pkl file
            model: str='random_forest', # model to tune. current options are 'random_forest', 'lasso', 'lightGBM'
	    params_path='best_params.pkl', #.pkl file
	    trials_path='trials.csv', #.csv file
	    n_trials=50,
	    random_state=42,
	    loss_func=mse_loss,
	    n_jobs=4,
	    cv_folds=5,
	    subsample_train=True,
	    timeout=3600 # stop tuning after one hour
	):

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

    X_train_copy = X_train.copy()

    # subsample from train set if train set is large to reduce runtime
    if subsample_train == True:
	X_train_copy = X_train_copy.sample(frac=0.25)
	    
    # Run optimization
    y_train = np.ravel(y_train)
    if prev_trials < n_trials:
        if model == 'random_forest':
            study.optimize(lambda trial: rf_reg_objective(trial, X_train_copy, y_train, random_state=random_state, loss_func=loss_func, n_jobs=n_jobs, cv_folds=cv_folds), n_trials=(n_trials-prev_trials), timeout=timeout)
        elif model == 'lasso':
            study.optimize(lambda trial: lasso_objective(trial, X_train_copy, y_train, random_state=random_state, cv_folds=cv_folds), n_trials=(n_trials-prev_trials), timeout=timeout)
        elif model == 'lightGBM':
            study.optimize(lambda trial: lightGBM_objective(trial, X_train_copy, y_train, random_state=random_state, cv_folds=cv_folds), n_trials=(n_trials-prev_trials), timeout=timeout)
	
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

def lgb_train_test_write(X_train,
			 X_test,
			 y_train,
			 y_test,
			 meta_train,
			 meta_test,
			 params_path='params_path.pkl',
			 model_dir='model',
			 proc_data_dir='data',
			 model_id='default',
			 log_label=True):

    with open(params_path, 'rb') as f:
         hyperparams=pickle.load(f)

    model = lgb.LGBMRegressor(**hyperparams)
    model.fit(X_train, y_train)

    # write model
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))

    # generate predictions
    y_pred = model.predict(X_test)

    # align indices
    meta_test = meta_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    results_df = pd.DataFrame(meta_test)
    if log_label == True:
        results_df['y_true_' + model_id] = [math.exp(x) for x in y_test]
        results_df['y_pred_' + model_id] = [math.exp(x) for x in y_pred]
    else:
        results_df['y_true_' + model_id] = y_test
        results_df['y_pred_' + model_id] = y_pred
    results_df['ratio_' + model_id] = results_df['y_pred_' + model_id]/results_df['y_true_' + model_id]
    results_df['model_id'] = model_id

    # write predictions and metadata
    results_df.to_csv(os.path.join(proc_data_dir, model_id + '_preds.csv'), index=False)

    return results_df
    
def rf_train_test_write(X_train, 
			X_test, 
			y_train, 
			y_test, 
			meta_train, 
			meta_test, 
			params_path='params_path.pkl',
			model_dir='model',
			proc_data_dir='data',
		        model_id='default',
		        log_label=True):

    """
    Train model using optimal hyperparams
    Write out predictions along with ground truth and metadata to specified directory

    inputs:
    -X_train: dataframe of training data features
    -X_test: dataframe of test data features
    -y_train: dataframe of training data labels
    -y_test: dataframe of test data labels
    -meta_train: metadata from training set
    -meta_test: metatdata from test set
    -params_path: .pkl file where the model's hyperparameters live
    -proc_data_dir: directory where the output should live
    -model_id: string indicating the model being run
    -log_label: indicates whether label was log-transformed for training, and therefore
    labels and predictions should be exponentiated before writing results to memory. 

    outputs:
    -results_df: dataframe, written to proc_data_dir, which contains
    sale price, predicted value, sales ratio, model_id, and desired metadata
    for each observation in the test set.
    also contains exponentiated labels and predictions if log_label == True.
    """

    with open(params_path, 'rb') as f:
         hyperparams=pickle.load(f)

    model = RandomForestRegressor(**hyperparams)
    model.fit(X_train, y_train)

    # write model
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))

    # gen preds
    y_pred = model.predict(X_test)

    # align indices
    meta_test = meta_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    results_df = pd.DataFrame(meta_test)
    if log_label == True:
        results_df['y_true_' + model_id] = [math.exp(x) for x in y_test]
        results_df['y_pred_' + model_id] = [math.exp(x) for x in y_pred]
    else:
        results_df['y_true_' + model_id] = y_test
        results_df['y_pred_' + model_id] = y_pred
    results_df['ratio_' + model_id] = results_df['y_pred_' + model_id]/results_df['y_true_' + model_id]
    results_df['model_id'] = model_id

    # write predictions and metadata
    results_df.to_csv(os.path.join(proc_data_dir, model_id + '_preds.csv'), index=False)

    return results_df
