import os
import yaml
import numpy as np
import pandas as pd

from typing import Union

class Setup:

    '''
    Class which sets up a model config and data for a specified
    county and model type. 
    
    '''

    def __init__(self,
             fips: str='00000', # 5-digit fips code, represented as a string.
             fips_county_crosswalk: str='../config/county_dict.yaml', # location of crosswalk between fips code and county name
             model_type: str='rf', # specifies type of model. examples: 'rf', 'lasso', 'lightGBM'
             study_label: str='ablation_study', # label for the type of study being run
             n_features: int=-1, # specify number of model features (for ablation study). if -1 or None, include all specified.
             feature_list: str='../config/full_feature_list.yaml', # config with full list of features of each type
             continuous: list=None, # list of continuous features to include. if None, defaults to list in feature_list
             categorical: list=None, # list of categorical features to include. if None, defaults to list in feature_list
             census: Union[str,list]=None,  # valid options are 'bg', 'tract' or list. if None, no census features included.
             label: str=None, # model label. if None, defaults to label specified in feature_list.
             log_label: bool=True, # toggle whether to apply log transformation to the label
             loss_func: str=None, # loss function we want to include in the config.
             base_config: str='../config/base_config.yaml' # base config for use as input to build_config()
             ):
        
        if fips:
            if isinstance(fips, str) and len(fips) == 5:
                self.fips = fips
            else:
                raise ValueError("fips must be a five-digit string.")
        else:
            raise ValueError('Must specify fips.')
          
        if model_type:
            if isinstance(model_type, str):
                self.model_type = model_type
            else:
                raise ValueError("model_type must be a string.")
        else:
            raise ValueError('Must specify model_type.')
          
        if study_label:
            if isinstance(study_label, str):
                self.study_label = study_label
            else:
                raise ValueError("study_label must be a string.")
        else:
            raise ValueError("Must specify study_label.")
        
        if feature_list:
            if isinstance(feature_list, str) and os.path.exists(feature_list):
                try:
                    with open(feature_list, 'r') as stream:
                        self.feature_list = yaml.safe_load(stream)
                except:
                    raise ValueError("Error trying to load feature_list as yaml.")
        else:
            self.feature_list = None

        if fips_county_crosswalk:
            if isinstance(fips_county_crosswalk, str) and os.path.exists(fips_county_crosswalk):
                try:
                    with open(fips_county_crosswalk, 'r') as stream:
                        self.fips_county_crosswalk = yaml.safe_load(stream)
                    self.county_name = self.fips_county_crosswalk[self.fips]
                except:
                    raise ValueError("Error trying to load fips_county_crosswalk as yaml.")
        else:
            self.fips_county_crosswalk = None
           
        if n_features:
            if isinstance(n_features, int) and (n_features == -1 or n_features > 0):
                self.n_features = n_features
            else:
                raise ValueError("n_features must be an integer that is either > 0 or equal to -1.")
        else:
            self.n_features = None

        if continuous:
            if isinstance(continuous, list) and all(isinstance(item, str) for item in continuous):
                self.continuous = continuous
            else:
                raise ValueError('continuous must be a list containing only strings.')
        else:
            try:
                self.continuous = self.feature_list['continuous']
            except:
                raise ValueError("Error trying to load continuous features from feature list.")

        if categorical:
            if isinstance(categorical, list) and all(isinstance(item, str) for item in categorical):
                self.categorical = categorical
            else:
                raise ValueError('categorical must be a list containing only strings.')
        else:
            try:
                self.categorical = self.feature_list['categorical']
            except:
                raise ValueError("Error trying to load categorical features from feature_list.")

        if census:
            if isinstance(census, list) and all(isinstance(item, str) for item in census):
                self.census = census
            elif census == 'bg':
                try:
                    self.census = self.feature_list['census_block_group']
                except:
                    raise ValueError("Error trying to load census block group features from feature_list.")
            elif census == 'tract':
                try:
                    self.census = self.feature_list['census_tract']
                except:
                    raise ValueError("Error trying to load census tract features from feature_list.")
            else:
                raise ValueError('census must equal "bg" or "tract", or be a list of strings indicating census features to include.')
        else:
            self.census = None

        if label:
            if isinstance(label, str):
                self.label = label
            else:
                raise ValueError('label must be str or None.')
        else:
            try:
                self.label = self.feature_list['label']
            except:
                raise ValueError("Error processing label from feature_list. ensure label is present in feature_list or specify label directly as string.")
            
        if log_label:
            if isinstance(log_label, bool):
                self.log_label = log_label
            else:
                raise ValueError('log_label must be boolean.')
            
        else:
            self.log_label = True

        if loss_func:
            if isinstance(loss_func, str):
                self.loss_func = loss_func
            else:
                raise ValueError('loss_func must be a string.')
        else:
            raise ValueError('must specify loss function loss_func')

        if base_config:
            if isinstance(base_config, str) and os.path.exists(base_config):
                try:
                    with open(base_config, 'r') as stream:
                        self.base_config = yaml.safe_load(stream)
                except:
                    raise ValueError("Error trying to load feature_list as yaml.")
            else:
                raise ValueError("base_config must be string and base config file must exist at location specified in string.")
        else:
            self.base_config = None


    def gen_paths(self, base_dir='/oak/stanford/groups/deho/proptax/models/'): 

        """
        generates paths required to store model data, logs, config, and ouput

        returns generated paths for later use, as these paths need to be passed to functions in preprocess and modeling utils.
        initially, they were stored directly in the config file.

        We can also lump this in to build_config()

        """

        # get name of county using fips-county crosswalk



        dir_list = ['hyperparams/studies', 
                    'hyperparams/samplers', 
                    'hyperparams/best_params', 
                    'hyperparams/trials',
                    'data',
                    'logs',
                    'encoders',
                    'config',
                    'model']

        directories = []
        
        for dir in dir_list:
            if not os.path.exists(os.path.join(base_dir, f"{self.county_name}_{self.model_type}_{self.study_label}", dir)):
                os.makedirs(os.path.join(base_dir, f"{self.county_name}_{self.model_type}_{self.study_label}", dir))
                print(f"Created directory: {os.path.join(base_dir, f"{self.county_name}_{self.model_type}_{self.study_label}", dir)}")
            else:
                print(f"Directory already exists: {os.path.join(base_dir, f"{self.county_name}_{self.model_type}_{self.study_label}", dir)}")

            directories.append(os.path.join(base_dir, f"{self.county_name}_{self.model_type}_{self.study_label}", dir))

        return directories
    
    def build_config(self, feature_order_path='../config/feature_order_2023.csv', base_dir='/oak/stanford/groups/deho/proptax/models/'):
        """
        builds config file according to specifications provided by user.

        includes option to dynamically set size of feature set according to 
        which features are present in the data for a given county.

        idea: features should be deleted in ascending order of availability across counties, i.e. features
        that are not present in very many counties are deleted first.
        
        
        """
        # Check if list of features ordered by most to least common within a county exists
        if os.path.exists(feature_order_path):
            try:
                feature_order = pd.read_csv(feature_order_path)
            except:
                raise ValueError("Error trying to load feature_order_path as csv.")
                
        else: 
            ## If not, create and save a list

            ## Load data
            df = pd.read_csv(self.base_config['paths']['raw_path'])
            
            ## Subset data to all columns used for modeling
            ## This should always come from full_feature_list 
            ## (because it will become the default order for subsequent calls)
            with open('../config/full_feature_list.yaml', 'r') as stream:
                full_feature_list = yaml.safe_load(stream)
            df = (df[['fips'] + full_feature_list['continuous'] + full_feature_list['categorical']])

            ## Calculate # of counties that have at least share_non_null values for a given column
            fips = df.loc[:,'fips']
            df = df.drop(columns=['fips']).notna()
            df.loc[:,'fips'] = fips
            feature_order = df.groupby('fips').mean().reset_index()

            feature_order.to_csv(feature_order_path, index=False)
            feature_order = pd.read_csv(feature_order_path)

        ## Sort columns in order of the number of counties that have at least share_non_null values 
        features = (feature_order.drop(columns=['fips']) >= self.base_config['model_params']['share_non_null']).sum().reset_index().rename(columns={'index' : 'feature', 0 : 'availability'})
        features = features.sort_values('availability', ascending=False)['feature'].to_list()

        ## Select only features present in specified fips
        fips_features = feature_order[feature_order['fips'] == int(self.fips)].drop(columns=['fips']).reset_index(drop=True).T.reset_index().rename(columns={ 'index' : 'feature', 0 : 'availability'})
        fips_features = fips_features[fips_features['availability'] >= self.base_config['model_params']['share_non_null']]['feature'].to_list()

        ## Order features that are present in the fips
        feature_order = [x for x in feature_order if x in fips_features]

        ## Subset features to specified number
        if self.n_features > len(feature_order):
            raise ValueError("More features requested than available in fips.")

        elif self.n_features == -1:
            feature_order = feature_order

        else: 
            feature_order = feature_order[:self.n_features]

        ## Split vars into categorical and continuous (continuous includes census vars)
        ## This step also restricts to the categorical and continuous lists provided in Setup(), if applicable
        categorical = [x for x in feature_order if x in self.feature_list['categorical']]
        continuous = [x for x in feature_order if x in self.feature_list['continuous']]

        ## Generate dir_list
        dir_list = self.gen_paths(base_dir)

        ## Generate model ID
        model_id = f"{self.fips}_{self.county_name}_{self.model_type}_{self.study_label}_{self.n_features}_features"

        config = {'rand_state' : self.base_config['model_params']['rand_state'], 
                  'test_size' : self.base_config['model_params']['test_size'], 
                  'cv_folds' : self.base_config['model_params']['cv_folds'], 
                  'mice_iters' : self.base_config['model_params']['mice_iters'], 
                  'max_iters' : self.base_config['model_params']['max_iters'], 
                  'n_trials' : self.base_config['model_params']['n_trials'], 
                  'n_jobs' : self.base_config['model_params']['n_jobs'], 
                  'share_non_null' : self.base_config['model_params']['share_non_null'], 
                  'min_samples_leaf' : self.base_config['model_params']['min_samples_leaf'], 
                  'smoothing' :  self.base_config['model_params']['smoothing'],
                  'write_encoding_dict' : self.base_config['model_params']['write_encoding_dict'], 

                  'log_label' : self.log_label, 

                  'dir_list' : dir_list, 
                  'raw_path': self.base_config['paths']['raw_path'],

                  'study_dir': f"{dir_list[0]}/studies",
                  'sampler_path': f"{dir_list[1]}/sampler.pkl",
                  'params_path': f"{dir_list[2]}/best-params.pkl",
                  'trials_path': f"{dir_list[3]}/trials.pkl",
                  'model_dir': dir_list[8],
                  'proc_data_dir': dir_list[4],
                  'log_dir': dir_list[5],
                  'encoding_path': dir_list[6],

                  'fips': self.fips,
                  'county_name': self.county_name,
                  'model_id': model_id,
                  'model_type': self.model_type,
                  'meta': self.base_config['features']['meta'],

                  'time': self.base_config['features']['time'], 
                  'label': self.label,
                  'binary': self.base_config['features']['binary'],
                  'continuous': continuous,
                  'categorical': categorical}

        config_path = dir_list[7]

        with open(f"{config_path}/{model_id}.yaml", 'w') as stream:
            yaml.dump(config, stream)

