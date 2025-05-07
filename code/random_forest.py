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
sampler_path = out['sampler_path']
params_path = out['params_path']
raw_path = out['raw_path']
out_path = out['out_path']
fips = out['fips']

label = out['label']
continuous = out['continuous']
features = out['continuous'] + out['missing_ind'] + out['binary']

# load raw data
df = pd.read_csv(raw_path)

# subset to single county
df = df.loc[df.fips == fips]

# define features, label
X = df[features]
y = df[label]

# preproc: impute missings, normalize continuous vars
X = impute_and_normalize(X, continuous, random_state=rand_state, mice_iters=mice_iters)


