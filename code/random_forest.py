import logging
import miceforest as mf
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
import yaml

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
from preprocess import *

# change working directory to this script's location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# load config
with open('../config/config.yaml', 'r') as stream:
        out = yaml.safe_load(stream)

# assign parameters, paths, and feature names according to config
random_state = out['rand_state']
test_size = out['test_size']
cv_folds = out['cv_folds']
mice_iters = out['mice_iters']
n_trials = out['n_trials']
n_jobs = out['n_jobs']
share_non_null = out['share_non_null']
min_samples_leaf = out['min_samples_leaf']
smoothing = out['smoothing']
write_encoding_dict=out['write_encoding_dict']

# paths
study_name = out['study_name']
sampler_path = out['sampler_path']
params_path = out['params_path']
raw_path = out['raw_path']
out_path = out['out_path']
trials_path = out['trials_path']
log_dir = out['log_dir']
encoding_path = out['encoding_path']

fips = out['fips']
label = out['label']
continuous = out['continuous']
#binary = out['binary'] + out['missing_ind']
binary = out['binary']
categorical = out['categorical']
#features = out['continuous'] + out['missing_ind'] + out['binary']
features = out['continuous'] + out['binary']
meta = out['meta']

# load raw data
df = pd.read_csv(raw_path)
print('data loaded; ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
print('subsetting to county ' + str(fips))

# subset to single county
df = df.loc[df.fips == fips]
print(str(df.shape[0]) + ' rows remaining after county-level subset')

preproc = Preprocess(df.copy(),
		    label,
		    continuous,
		    binary,
		    categorical,
		    meta,
		    share_non_null=share_non_null,
		    random_state=random_state,
		    wins_pctile=1,
		    mice_iters=mice_iters,
		    log_dir=log_dir,
		    min_samples_leaf=min_samples_leaf,
		    smoothing=smoothing,
		    write_encoding_dict=write_encoding_dict,
		    encoding_path=encoding_path)

X_train, X_test, y_train, y_test, meta_train, meta_test, continuous, binary, categorical = preproc.run()
print(f'continuous_cols: {continuous}')
print(f'binary_cols: {binary}')
print(f'categorical_cols: {categorical}')

X_train.to_csv('/oak/stanford/groups/deho/proptax/results/processed_data/X_train.csv')
X_test.to_csv('/oak/stanford/groups/deho/proptax/results/processed_data/X_test.csv')
y_train.to_csv('/oak/stanford/groups/deho/proptax/results/processed_data/y_train.csv')
y_test.to_csv('/oak/stanford/groups/deho/proptax/results/processed_data/y_test.csv')
'''
# define features, label, meta
X = df[features].reset_index(drop=True)
y = df[label].reset_index(drop=True)
meta = df[meta].reset_index(drop=True)

print('Labels, features defined. Imputing missing values and normalizing continuous variables.')
# first: drop cols with high numbers of nulls, missings
X = subset_cols(X, n_non_null=100)

# preproc: impute missings, normalize continuous vars
X = impute_and_normalize(X, continuous, random_state=rand_state, mice_iters=mice_iters)

print('data preprocessed; creating train-test splits')
split_fn = get_data_splitter(X, y, meta, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test, meta_train, meta_test = split_fn()

print('tuning model hyperparams')
tune_model(X_train, 
           y_train,
            study_name=study_name,
            load_if_exists=True,
            sampler=TPESampler(seed=rand_state),
            sampler_path=sampler_path, #.pkl file
            params_path=params_path, #.pkl file
            trials_path=trials_path, #.csv file
            n_trials=n_trials,
            test_size=test_size,
	    random_state=rand_state,
	    loss_func=mpe2_loss,
            n_jobs=n_jobs,
	    cv_folds=cv_folds)

print('optimal params selected; training and testing model')
rf_train_test_write(X_train, 
                    X_test, 
                    y_train, 
                    y_test, 
                    meta_train, 
                    meta_test, 
                    params_path=params_path, 
                    out_path=out_path)
'''
