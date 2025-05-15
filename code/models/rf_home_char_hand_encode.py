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

# change working directory to this script's location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

sys.path.insert(0, '..')
from modeling_utils import *
from preprocess import *

# load config
with open('../../config/rf_home_char_hand_encode_config.yaml', 'r') as stream:
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
study_dir = out['study_dir']
sampler_path = out['sampler_path']
params_path = out['params_path']
raw_path = out['raw_path']
proc_data_dir = out['proc_data_dir']
trials_path = out['trials_path']
log_dir = out['log_dir']
encoding_path = out['encoding_path']

fips = out['fips']
label = out['label']
continuous = out['continuous']
binary = out['binary']
categorical = out['categorical']
features = out['continuous'] + out['binary'] + out['categorical']
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

# run preprocessor
X_train, X_test, y_train, y_test, meta_train, meta_test, continuous, binary, categorical = preproc.run(target_encode=False)

# write pre-processed train and test sets
for key, value in {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'meta_train': meta_train, 'meta_test': meta_test}.items():
    value.to_csv(os.path.join(proc_data_dir, f'{key}.csv'))
				  
# tune model hyperparams
tune_model(X_train, 
           y_train,
            study_name=study_dir, # todo - need to update tune_model to reflect changes in config setup
            load_if_exists=True,
            sampler=TPESampler(seed=random_state),
            sampler_path=sampler_path, #.pkl file
            params_path=params_path, #.pkl file
            trials_path=trials_path, #.csv file
            n_trials=n_trials,
            test_size=test_size,
	    random_state=random_state,
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
                    proc_data_dir=proc_data_dir)
