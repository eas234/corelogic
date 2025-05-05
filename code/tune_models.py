import logging
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

import optuna
from optuna.samplers import TPESampler, RandomSampler

def mpe2_loss(y_true, 
              y_pred):
    '''
    Mean Percentage Error Squared loss function
    for use as alternative to MSE in model dev.
    '''
                  
    loss = (((y_true - y_pred)**2 / (y_true)**2).sum())/len(y_true)
                  
    return loss

def rf_reg_objective(trial, 
              X, 
              y, 
              test_size=0.2, 
              random_state=42, 
              loss_func=mpe2_loss, 
              n_jobs=4,
              cv_folds=5):

    '''
    Ranndom forest regressor objective for use in optuna 
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

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = RandomForestRegressor(random_state=random_state, **params)
    scorer = make_scorer(loss_func)
    cv_score = cross_val_score(model, X_train, y_train, scoring=scorer, n_jobs=n_jobs, cv=cv_folds)
    mean_cv_accuracy = cv_score.mean()
                  
    return mean_cv_accuracy

def tune_model(objective, 
            study_name="example-study",
            storage_name="sqlite:///{}.db".format(study_name),
            load_if_exists=True,
            sampler=TPESampler(seed=42),
            sampler_path='sampler.pkl', #.pkl file
            params_path='best_params.pkl', #.pkl file
            trials_path='trials.csv', #.csv file
            n_trials=20
              ):

    '''
    Model tuner
    '''

    # check if sampler is saved from a previous run and load if it is
    if os.path.isfile(sampler_path):
        with open(sampler.path, "rb") as f:
            restored_sampler = pickle.load(f)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=load_if_exists,
            sampler=restored_sampler
        )
    
    else:
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage_name, 
            load_if_exists=load_if_exists, 
            sampler=sampler
        )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Save the sampler
    with open(sampler_path, "wb") as fout:
        pickle.dump(study.sampler, fout)

    # Save the best parameters
    with open(params_path, 'wb') as fout:
        pickle.dump(study.best_params, fout)

    # Output trials DataFrame
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(trials_path)
