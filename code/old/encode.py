import datetime
import logging
import numpy as np
import os
import pandas as pd
import yaml

from category_encoders import *
from sklearn.model_selection import train_test_split


class Encode:

    """
    Contains methods to encode categorical variables.

    Methods include:
    """

    def __init__(self, 
                 data: pd.DataFrame, # dataframe with features to encode
                 label: str=None, # string indicating target or label in data
                 features: list=None, # list indicating features in data
                 categorical_cols: list=None, # categorical cols that need encoding
                 meta_cols: list=None, # meta features that should not be modified 
                 random_state: int=42,
                 test_size: float=0.2,
                 log_dir: str='logs' # filepath for logger
                 ):

        # set log filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_filename = f"encode_log_{timestamp}.log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers if re-instantiated
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Logger initialized.")

        # private attributes
        self._data = data

        if label:
            self._label = label
        else:
            self.logger.warning("No label specified. Must specify label.")
            raise ValueError("No label specified. Must specify label.")
        
        if features:
            self._features = features
        else:
            self.logger.warning("No features specified. Must specify at least one feature.")
            raise ValueError("No features specified. Must specify at least one feature.")

        if categorical_cols:
            self._categorical_cols = categorical_cols
        else:
            self.logger.warning("Must specify at least one categorical column.")
            raise ValueError("Must specify at least one categorical column.")
        
        self._meta_cols = meta_cols

        # protected attributes
        self.__random_state = random_state
        if test_size > 0 and test_size < 1:
            self.__test_size = test_size
        else:
            self.logger.warning("test_size must be a float between 0 and 1")
            raise ValueError("test_size must be a float between 0 and 1")

    def train_test_split(self, return_items: bool=False):

        """
        Generates train-test split attributes
        using sklearn's train_test_split.

        Inputs:
        - self.__random_state: for repoducibility
        - self.__test_size: desired size of test set as fraction of self_data
        - self._data: full dataset including features, labels, and meta cols
        - self._features: features to include in X_train, X_test
        - self._label: label to include in y_train, y_test
        - self._meta_cols: meta features like property ids that should be 
            split using same indices as X and y but which are not used in 
            model development
        - return_items: if true, returns test-train split 

        Returns:
        self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test
        as attributes of Encode() object.
        """

        self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test = train_test_split(self._data[self._features], 
                                                                                                self._data[self._label], 
                                                                                                self._data[self._meta_cols], 
                                                                                                test_size=self.__test_size, 
                                                                                                random_state=self.__random_state)
        
        self.logger.info("X_train, X_test, y_train, y_test, meta_train, meta_test now stored as attributes of Encode() object")
    
        if return_items:
            return self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test

    def target_encode(self, 
                      min_samples_leaf: int=20, # smoothing parameter, see docstring
                      smoothing: int=10, # smoothing parameter, see docstring
                      inplace: bool=True, # governs whether encoding happens inplace
                      write_encoding_dict: bool=True, # if true, writes dictionary mapping encodings to categories
                      encoding_path: str=None # path to write encodings
                      ):

        """
        Encodes categorical variables specified in self._categorical_cols
        using category_encoder's TargetEncoder().

        For each value v of a categorical variable c, the model outputs a
        weighted average between the the prior and posterior of the outcome y.
        The prior is the average of y across the entire training set, while 
        the posterior is the average of y among observations with c=v.

        Formally:

        v_encoded = lambda(n)*posterior + (1-lambda(n))*prior

        Smoothing is governed by the following sigmoid function:

        lambda(n) = 1 / (1 + e ^ ( (n-k) / f ))

        When k=n, lambda(n) = 0.5
        As f approaches infinity, lambda(n) --> 1 and v_encoded --> posterior

        Inputs:
        -cols=self._categorical_cols, columns to encode
        -min_samples_leaf: k
        -smoothing: f
        -write_encoding_dict: boolean governing whether method writes encoding dictionary
        mapping encodings to category values as .yaml file
        -encoding_path: path where encoding dict is written 

        Outputs:
        - if inplace, updates self._X_test and self._X_train with encoded categorical vars
        - if not inplace, returns copies of X_test and X_train with encoded categorical vars

        """

        self.logger.info("Encoding categoricals using target_encode()")
        enc = TargetEncoder(cols=self._categorical_cols,
                             min_samples_leaf=min_samples_leaf, 
                             smoothing=smoothing).fit(self.X_train, self.y_train)
        self.logger.info("Encoder fitted")

        if write_encoding_dict and not encoding_path:
            self.logger.warning("encoding_path must be provided if write_encoding_dict = True.")
            raise ValueError("encoding_path must be provided if write_encoding_dict = True.")
        if write_encoding_dict:
            encoding_dicts={}

            for i in range(len(enc.ordinal_encoder.mapping)):
                col = enc.ordinal_encoder.cols[i]
                mapping_series = enc.ordinal_encoder.mapping[i]['mapping']
            
            # Create the combined dictionary
                combined_dict = {
                    category: enc.mapping[col][code]
                    for category, code in mapping_series.items()
                }
                combined_dict['UNSEEN'] = enc.mapping[col][-1]
                encoding_dicts[col]=combined_dict

            with open(os.path.join(encoding_path, 'encodings.yaml'), 'w') as f:
                yaml.dump(encoding_dicts, f)
            
            self.logger.info(f'Encodings written to {encoding_path} as encodings.yaml')
        
        if inplace:
            self.logger.info("Encoding categoricals inplace")
            self.X_train = enc.transform(self.X_train)
            self.X_test = enc.transform(self.X_test)

        else:
            self.logger.info("Encoding categoricals on copies of X_train, X_test")
            X_train = enc.transform(self.X_train.copy())
            X_test = enc.transform(self.X_test.copy())
            return X_train, X_test

    def kmeans_encode():
        """
        Will develop later.
        """
        raise NotImplementedError("kmeans_encode() is not yet implemented")

    def run(self, 
            method: str=None, # encoding method to run
            return_items: bool=False, # whether to return test/train sets
            **kwargs
            ):

        """
        Wrapper.

        Inputs: 
        - method: encoder to use
        - return_items: bool, indicates whether to return test/train objects
        - **kwargs: keyword arguments for chosen method
        """

        if method == "target_encode":
            method_to_call = self.target_encode
        elif method == "kmeans_encode":
            method_to_call = self.kmeans_encode
        else:
            raise ValueError(f"Unknown method: {method}")
    
        method_to_call(**kwargs)
        
        if return_items == True:
            return self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test
            



