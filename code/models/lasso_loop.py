
import numpy as np
import os
import pandas as pd
import sys
import yaml

sys.path.insert(0, '..')
from preprocess import Preprocess
from modeling_utils import mae_loss, lasso_objective, tune_model, lasso_train_test_write

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

## change working directory to this script's location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

## load lasso config file
with open('../../config/lasso_config.yaml', 'r') as stream:
    out = yaml.safe_load(stream)

## load paths, params, and variable lists from config

#paths
data_path = out['paths']['data_path']
feature_order_path = out['paths']['feature_order_path']
fips_path = out['paths']['fips_path']
results_path = out['paths']['results_path']
log_path = out['paths']['log_path']
sampler_path = out['paths']['sampler_path']
params_path = out['paths']['params_path']
trials_path = out['paths']['trials_path']

#params
share_non_null = out['model_params']['share_non_null']
random_state = out['model_params']['random_state']
wins_pctile = out['model_params']['wins_pctile']
log_label = out['model_params']['log_label']
n_trials = out['model_params']['n_trials']
loss_func = out['model_params']['loss_func']

#varbs
categorical_full = out['features']['categorical']
continuous_full = out['features']['continuous']
meta = out['features']['meta']
label = out['features']['label']
sale_date_col = out['features']['sale_date']

## load in full data
print('loading data')
df = pd.read_csv(data_path)
#df = pd.read_csv(data_path, nrows=1000000) # subsample for testing

print('data loaded')

# data cleaning
df = df[df.MULTI_OR_SPLIT_PARCEL_CODE.isnull()]
df = df[~df.fips.isnull()]
#df[sale_date_col] = pd.to_datetime(df[sale_date_col], errors='coerce')
#df[sale_date_col] = df[sale_date_col].dt.strftime("%Y%m%d").astype(int)
df.fips = [int(x) for x in df.fips]

print('data cleaned')

## get list of fips to loop through; determine how many fips have already been processed
fips = pd.read_csv(fips_path)
fips_list = [int(x) for x in fips.fips.unique().tolist()]

if os.path.exists(results_path):
    results = pd.read_csv(results_path)
    completed_fips = [int(x) for x in results.fips]
    remaining_fips = set(fips_list) - set(completed_fips)
    print(f"{len(remaining_fips)} counties remaining out of {len(fips_list)}")
else:
    results = pd.DataFrame(columns=meta+['y_true', 'y_pred', 'ratio', 'model_id'])
    print(f"{len(fips_list)} counties remaining out of {len(fips_list)}")

## load feature order list
feature_order = pd.read_csv(feature_order_path)

## Sort columns in order of the number of counties that have at least share_non_null values 
features = (feature_order.drop(columns=['fips']) >= share_non_null).sum().reset_index().rename(columns={'index' : 'feature', 0 : 'availability'})
features = features.sort_values('availability', ascending=False)['feature'].to_list()

for fips in fips_list:
    print(f'starting fips {fips}')

    # clear paths
    for temp_path in [sampler_path, params_path, trials_path, 'lasso_loop.db']:
        if os.path.exists(temp_path):
            os.remove(temp_path)
	
    # subset to sales from fips
    data = df[df.fips == fips].copy()

    ## Select only features present in specified fips
    fips_features = feature_order[feature_order['fips'] == int(fips)].drop(columns=['fips']).reset_index(drop=True).T.reset_index().rename(columns={ 'index' : 'feature', 0 : 'availability'})
    fips_features = fips_features[fips_features['availability'] >= share_non_null]['feature'].to_list()

    ## Order features that are present in the fips
    feature_order_list = [x for x in features if x in fips_features]

    ## designate categorical and continuous
    categorical = [x for x in feature_order_list if x in categorical_full]
    continuous = [x for x in feature_order_list if x in continuous_full]
    binary = []

    ###

    preproc = Preprocess(data.copy(),
			    label,
			    continuous,
			    binary,
			    categorical,
			    meta,
                sale_date_col,
			    share_non_null=share_non_null,
			    random_state=random_state,
			    wins_pctile=1,
			    log_label=log_label,
                log_dir=log_path
                )
    
    X_train, X_test, y_train, y_test, meta_train, meta_test, continuous, binary, categorical = preproc.run(target_encode=False, 
                                                                                                           one_hot=True, 
                                                                                                           drop_lowest_ratios=True,
																										   gen_time_vars=True
                                                                                                           )

    ## tune, train, and write output from full-feature model

    # tune
    tune_model(X_train, 
	            y_train,
 	            study_name='lasso_loop',
                load_if_exists=True,
                sampler=TPESampler(seed=42, n_startup_trials=20, multivariate=True),
                sampler_path=sampler_path, #.pkl file
                params_path=params_path, #.pkl file
                trials_path=trials_path, #.csv file
                geography=None,
                n_trials=n_trials,
                random_state=random_state,
                loss_func=loss_func,
                subsample_train=False, # to do: add toggle in config
                model='lasso',
                write_output=True,
                )


    # train

    output = lasso_train_test_write(X_train, 
			X_test, 
			y_train, 
			y_test, 
			meta_test, 
			params_path=params_path,
            model_id='max_features',
            log_label=True,
            write_output=False)
    
    # write
    results = pd.concat([results, output])
    results.to_csv(results_path, index=False)

    # delete temporary paths created during model tuning
    for temp_path in [sampler_path, params_path, trials_path, 'lasso_loop.db']:
        if os.path.exists(temp_path):
            os.remove(temp_path)
         
    ## tune, train, and write output for 3-feature model

    X_train = X_train[X_train.columns.tolist()[:3]]
    X_test = X_test[X_train.columns.tolist()[:3]]

    # tune
    tune_model(X_train, 
	            y_train,
 	            study_name='lasso_loop',
                load_if_exists=False,
                sampler=TPESampler(seed=42, n_startup_trials=20, multivariate=True),
                sampler_path=sampler_path, #.pkl file
                params_path=params_path, #.pkl file
                trials_path=trials_path, #.csv file
                geography=None,
                n_trials=n_trials,
                random_state=random_state,
                loss_func=loss_func,
                subsample_train=False, # to do: add toggle in config
                model='lasso',
                write_output=True,
                )
    
    # train

    output = lasso_train_test_write(X_train, 
			X_test, 
			y_train, 
			y_test, 
			meta_test, 
			params_path=params_path,
            model_id='3_features',
            log_label=True,
            write_output=False)
    
    # write
    results = pd.concat([results, output])
    results.to_csv(results_path, index=False)

    # delete temporary paths created during model tuning
    for temp_path in [sampler_path, params_path, trials_path, 'lasso_loop.db']:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"{fips} complete.")
