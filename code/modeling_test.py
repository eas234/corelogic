import logging
import miceforest as mf
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys

import optuna
from optuna.samplers import TPESampler, RandomSampler

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

from modeling_utils import *

from sklearn.datasets import load_diabetes

# Load dataset
diabetes = load_diabetes()

# Features as DataFrame
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Target as Series
y = pd.Series(diabetes.target, name="target")

# Meta as series
meta = np.ones(len(y))

print('data loaded; simulating missing data')

# ampute columns to simulate missing data
amputed_df = mf.ampute_data(X,
    		perc=0.3,                  # Proportion of cells to make missing
    		random_state=42
)

print('missing data simulated; preprocessing')

X = impute_and_normalize(amputed_df, amputed_df.columns.tolist(), random_state=42, mice_iters=3)

print('data preprocessed; creating train-test splits')
split_fn = get_data_splitter(X, y, meta, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test, meta_train, meta_test = split_fn()

print('tuning model hyperparams')
tune_model(X_train, 
           y_train,
            study_name="runs/modeling-test",
            load_if_exists=True,
            sampler=TPESampler(seed=42),
            sampler_path='runs/sampler.pkl', #.pkl file
            params_path='runs/modeling_test_best_params.pkl', #.pkl file
            trials_path='runs/trials.csv', #.csv file
            n_trials=20,
            test_size=0.2,
	        random_state=42,
	        loss_func=mpe2_loss,
	        n_jobs=4,
	        cv_folds=5)

print('optimal params selected; training and testing model')

rf_train_test_write(X_train, 
                    X_test, 
                    y_train, 
                    y_test, 
                    meta_train, 
                    meta_test, 
                    params_path='runs/modeling_test_best_params.pkl', 
                    outdir='preds/')
