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

# change working directory to this script's location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# load config
with open('../config/config.yaml', 'r') as stream:
        out = yaml.safe_load(stream)

# assign parameters, paths, and feature names according to config
rand_state = out['rand_state']
test_size = out['test_size']
cv_folds = out['cv_folds']
mice_iters = out['mice_iters']
n_trials = out['n_trials']
study_name = out['study_name']
sampler_path = out['sampler_path']
params_path = out['params_path']
raw_path = out['raw_path']
out_path = out['out_path']
fips = out['fips']

label = out['label']
continuous = out['continuous']
features = out['continuous'] + out['missing_ind'] + out['binary']
meta = out['id'] + out['benchmark']

# load raw data
df = pd.read_csv(raw_path)
print('data loaded; ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
print('subsetting to county ' + str(fips))

# subset to single county
df = df.loc[df.fips == fips]
print(str(df.shape[0]) + ' rows remaining after county-level subset')

# define features, label
X = df[features]
y = df[label]

print('Labels, features defined. Imputing missing values and normalizing continuous variables.')
# preproc: impute missings, normalize continuous vars
X = impute_and_normalize(X, continuous, random_state=rand_state, mice_iters=mice_iters)

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


