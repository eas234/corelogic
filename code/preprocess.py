import datetime
import logging
import miceforest as mf
import numpy as np
import os
import pandas as pd
import yaml

from category_encoders import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocess:
    """
    Class which contains methods to preprocess data.

    Methods include:
    
    - Setters and getters for all class attributes
    - drop_null_labels(): drops rows where label is null
    - drop_single_value_cols(): drops columns that have only one value
    - drop_mostly_null_cols(): drops columns that have fewer than share_non_null non-null values
    - winsorize_continuous(): winsorizes continuous features at wins_pctile and 100-wins_pctile
    - winsorize_labels(): winsorizes labels at wins_pctile and 100-wins_pctile
    - one_hot(): one-hot encodes categorical variables
    - mice_impute(): imputes missing values using miceforest
    - normalize_continuous_cols(): normalizes continuous variables using sklearn StandardScaler()
    - normalize_binary_cols(): same as above but for binary columns.
    
    """
    def __init__(self, 
                 data: pd.DataFrame, # dataframe to preprocess
                 label: str=None, # string indicating model target or label
                 continuous_cols: list=None, # list of continuous features
                 binary_cols: list=None, # list of binary features
                 categorical_cols: list=None, # list of categorical features
                 meta_cols: list=None, # columns of metadata that methods should not modify
                 share_non_null: float=0.25, # minimum share of non-null values required in each column
                 random_state: int=42, # for reproducibility
                 wins_pctile: int=1, # percentile at which data are winsorized (symmetric)
                 mice_iters: int=3, # n_iters for miceforest imputer
                 test_size: float=0.2, # desired size of test set
                 log_dir: str='logs', # logger filepath
                 min_samples_leaf: int=20, # param for target_encode()
                 smoothing: int=10, # param for target_encode()
                 write_encoding_dict: bool=False, # whether to write encoding dict from target_encode()
                 encoding_path: str=None # file location of encoding dict from target_encode()
                 ):
        
        """
        Initialize Preprocessor class and configure logging.
        """

        # set log filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"preprocess_log_{timestamp}.log"
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
        self._label = label or ''
        self._continuous_cols = continuous_cols or []
        self._binary_cols = binary_cols or []
        self._categorical_cols = categorical_cols or []
        self._meta_cols = meta_cols or []
        
        
        # protected attributes
        self.__share_non_null = share_non_null
        self.__random_state = random_state
        self.__wins_pctile = wins_pctile
        self.__mice_iters = mice_iters
        self.__test_size = test_size

        if isinstance(min_samples_leaf, int) and min_samples_leaf > 0:
            self.__min_samples_leaf = min_samples_leaf
        else:
            self.logger.warning("min_samples_leaf must be positive integer")
            raise TypeError("min_samples_leaf must be positive integer")
        
        if isinstance(smoothing, int) and smoothing > 0:
            self.__smoothing = smoothing
        else:
            self.logger.warning("smoothing must be positive integer")
            raise TypeError("smoothing must be positive integer")
        
        self.__write_encoding_dict = write_encoding_dict
        self.__encoding_path = encoding_path or ""

        self.logger.info("Preprocess class initialized.")

    ### setters and getters

    # input dataframe
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data

    # label
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, new_label):
        if isinstance(new_label, str):
            self._label = label
        else:
            self.logger.error("label must be a string")
            raise ValueError("label must be a string")
    
    # continuous features
    @property
    def continuous_cols(self):
        return self._continuous_cols
    
    @continuous_cols.setter
    def continuous_cols(self, new_continuous):
        if isinstance(new_continuous, list):
            self._continuous_cols = new_continuous
        else:
            self.logger.error("continuous_cols must be a list")
            raise ValueError("continuous_cols must be a list")

    # binary features
    @property
    def binary_cols(self):
        return self._binary_cols
    
    @binary_cols.setter
    def binary_cols(self, new_binary):
        if isinstance(new_binary, list):
            self._binary_cols = new_binary
        else:
            self.logger.error("binary_cols must be a list")
            raise ValueError("binary_cols must be a list")

    # categorical features
    @property
    def categorical_cols(self):
        return self._categorical_cols
    
    @categorical_cols.setter
    def categorical_cols(self, new_categorical):
        if isinstance(new_categorical, list):
            self._categorical_cols = new_categorical
        else:
            self.logger.error("categorical_cols must be a list")
            raise ValueError("categorical_cols must be a list")

    # meta features
    @property
    def meta_cols(self):
        return self._meta_cols

    @meta_cols.setter
    def meta_cols(self, new_meta):
        if isinstance(new_meta, list):
            self._meta_cols = new_meta
        else:
            self.logger.error("meta_cols must be a list")
            raise ValueError("meta_cols must be a list")
    
    # min non-null values
    @property
    def share_non_null(self):
        return self.__share_non_null
    
    @share_non_null.setter
    def share_non_null(self, new_n):
        if new_n >= 0 and new_n <= 1 and isinstance(new_n, float):
            self.__share_non_null = new_n
        else:
            self.logger.error("share_non_null must be a float between 0 and 1")
            raise ValueError("share_non_null must be a float between 0 and 1")

    # random state
    @property
    def random_state(self):
        return self.__random_state
    
    @random_state.setter
    def random_state(self, new_random_state):
        if new_random_state>=0 and isinstance(new_random_state, int):
            self.__random_state = new_random_state
        else:
            self.logger.error("random_state must be a non-negative integer")
            raise ValueError("random_state must be a non-negative integer")
        
    # winsorize percentile

    @property
    def wins_pctile(self):
        return self.__wins_pctile
    
    @wins_pctile.setter
    def wins_pctile(self, new_pctile):
        if new_pctile>=0 and isinstance(new_pctile, int):
            self.__wins_pctile = new_pctile
        else:
            self.logger.error("wins_pctile must be a non-negative integer")
            raise ValueError("wins_pctile must be a non-negative integer")

    # mice iters
    @property
    def mice_iters(self):
        return self.__mice_iters
    
    @mice_iters.setter
    def mice_iters(self, new_mice_iters):
        if new_mice_iters>0 and isinstance(new_mice_iters, int):
            self.__mice_iters = new_mice_iters
        else:
            self.logger.error("mice_iters must be a positive integer")
            raise ValueError("mice_iters must be a positive integer")
    
    def drop_null_labels(self, inplace: bool=True):

        """
        Drop observations where label is null. Observations are dropped at the row 
        level.

        If inplace==True, observations are dropped in place and self._data indices
        are reset after drop.

        Else, method returns copy of self._data called processed_data where 
        observations with missing labels have been dropped, and indices are 
        reset after drop.
        """

        before = len(self._data)
        if inplace==True:
            self._data.dropna(subset=[self._label], axis=0, inplace=True)
            self._data = self._data.reset_index(drop=True)
            self.logger.info(f"Dropped {before - len(self._data)} rows with null labels out of {before} total rows.")
        else:
            processed_data = self._data.dropna(subset=[self._label], axis=0, inplace=False)
            processed_data = processed_data.reset_index(drop=True)
            self.logger.info(f"Dropped {before - len(processed_data)} rows with null labels out of {before} total rows.")
            return processed_data

    def drop_single_value_cols(self, inplace: bool=True):

        """
        Drop colums in input data that have only one value.
        This includes columns that are fully null and columns that take on only
        one non-null value, e.g. 0 or 1.

        If inplace==True, columns are dropped in place and self._data indices are reset after drop.
        Method does not return anything.
        
        Else, method returns copy of self._data with dropped columns.
        """

        # identify single-value columns
        drop_cols = [col for col in self._data.columns if self._data[col].nunique(dropna=True) <= 1 and col not in self._meta_cols]

        # drop single-value columns if found
        if drop_cols:
            self.logger.warning(f"Warning: {len(drop_cols)} columns have only one value. Dropping {drop_cols}")
        else:
            self.logger.info("No single-value columns to drop.")
            pass

        if inplace==True:
            self._data = self._data.drop(columns=drop_cols)
            self._data = self._data.reset_index(drop=True)
            self.logger.info("Updating cols attributes to reflect drops")
            self._continuous_cols = [x for x in self._continuous_cols if x in self._data.columns]
            self._binary_cols = [x for x in self._binary_cols if x in self._data.columns]
            self._categorical_cols = [x for x in self._categorical_cols if x in self._data.columns]
            
        else:
            processed_data = self._data.copy()
            processed_data = processed_data.drop(columns=drop_cols)
            processed_data = processed_data.reset_index(drop=True)
            return processed_data

    def drop_mostly_null_cols(self, inplace: bool=True):

        """
        Drop columns in input data that have fewer than share_non_null 
        non-null values. 

        If inplace==True, columns are dropped in place and self._data indices are reset after drop.
        Method does not return anything.

        Else, method returns copy of self._data with dropped columns.
        """

        # identify mostly null cols
        drop_cols = [col for col in self._data.columns if self._data[col].notnull().sum() <= int(np.floor(self.__share_non_null*self._data.shape[0])) and col not in self._meta_cols]

        if drop_cols:
            self.logger.warning(f'Warning: {len(drop_cols)} columns have fewer than {self.__share_non_null*100} percent non-null values. Dropping {drop_cols}')
        else:
            self.logger.info('No mostly null columns to drop.')
            pass

        if inplace==True:
            self._data.drop(columns=drop_cols, inplace=True)
            self._data.reset_index(drop=True, inplace=True)
            self.logger.info("Updating cols attributes to reflect drops")
            self._continuous_cols = [x for x in self._continuous_cols if x in self._data.columns]
            self._binary_cols = [x for x in self._binary_cols if x in self._data.columns]
            self._categorical_cols = [x for x in self._categorical_cols if x in self._data.columns]

        else:
            processed_data = self._data.copy()
            processed_data = processed_data.drop(columns=drop_cols)
            processed_data = processed_data.reset_index(drop=True)
            return processed_data

    def one_hot(self, inplace: bool=True):

        """
        Method that one-hot encodes categorical features.

        If inplace==True:
        - generates dummies for all columns in self._categorical_cols
        that are present in self._data
        - drops categorical columns inplace in self._data, replacing
        them with dummy values
        - sets self._categorical_cols to []
        - adds dummy columns to self._binary_cols

        Else:
        - generates dummies for self._categorical_cols that are present in
        self._data
        - adds these dummies to a copy of self._data called processed_data
        - drops categorical columns in processed_data
        - returns processed_data
        
        """
        # subset to categorical columns that are still present in data
        # after previous methods have been applied.

        categorical_cols = [x for x in self._categorical_cols if x in self._data.columns]

        dummies = pd.get_dummies(self._data[categorical_cols], drop_first=False)
    
        if inplace:
            self._data.drop(columns=categorical_cols, inplace=True)
            self._data[dummies.columns] = dummies
            self._binary_cols += dummies.columns.tolist()
            self._categorical_cols = []
        else:
            processed_data = self._data.drop(columns=categorical_cols, inplace=False)
            processed_data[dummies.columns] = dummies
            return processed_data
        
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

        self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test = train_test_split(self._data[self._binary_cols + self._categorical_cols + self._continuous_cols], 
                                                                                                self._data[self._label], 
                                                                                                self._data[self._meta_cols], 
                                                                                                test_size=self.__test_size, 
                                                                                                random_state=self.__random_state)
        
        self.logger.info("X_train, X_test, y_train, y_test, meta_train, meta_test now stored as attributes of preprocess() object")

    def _drop_problematic_cols_from_splits(self):
        """
        Drops columns that become problematic after train-test split.
        """
        
        # First identify columns that are bad in either split
        train_problem_cols = self._find_bad_columns(self.X_train)
        test_problem_cols = self._find_bad_columns(self.X_test)
        
        # Union of problematic columns across splits
        all_problem_cols = list(set(train_problem_cols) | set(test_problem_cols))
        
        if all_problem_cols:
            self.logger.warning(
                f"Dropping {len(all_problem_cols)} columns that became problematic after splitting: "
                f"{all_problem_cols}"
            )
            self.X_train = self.X_train.drop(columns=all_problem_cols)
            self.X_test = self.X_test.drop(columns=all_problem_cols)
            self._update_column_attributes()
    
    def _find_bad_columns(self, df):
        """Identifies columns that should be dropped from a dataframe"""
        return [
            col for col in df.columns
            if (df[col].nunique(dropna=True) <= 1) or  # Single-value columns
               (df[col].notnull().sum() < max(1, int(np.floor(self.__share_non_null*df.shape[0]))))  # Mostly null columns
        ]
    
    def _update_column_attributes(self):
        """Updates the column-type tracking attributes after dropping columns"""
        remaining_cols = set(self.X_train.columns)
        
        self._continuous_cols = [col for col in self._continuous_cols if col in remaining_cols]
        self._binary_cols = [col for col in self._binary_cols if col in remaining_cols]
        self._categorical_cols = [col for col in self._categorical_cols if col in remaining_cols]
        
        
    def target_encode(self, inplace: bool=True):

        """
        TODO: update self._continuous_cols to include encoded cols after encoding; empty out categorical cols

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
        -self.__min_samples_leaf: k
        -self.__smoothing: f
        -self.__write_encoding_dict: boolean governing whether method writes encoding dictionary
        mapping encodings to category values as .yaml file
        -self.__encoding_path: path where encoding dict is written 

        Outputs:
        - if inplace, updates self._X_test and self._X_train with encoded categorical vars
        - if not inplace, returns copies of X_test and X_train with encoded categorical vars

        """

        self.logger.info("Encoding categoricals using target_encode()")
        enc = TargetEncoder(cols=tuple(self.X_train[self._categorical_cols].columns),
                             min_samples_leaf=self.__min_samples_leaf, 
                             smoothing=self.__smoothing).fit(self.X_train, self.y_train)
        self.logger.info("Encoder fitted")

        if self.__write_encoding_dict:
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

            with open(os.path.join(self.__encoding_path, 'encodings.yaml'), 'w') as f:
                yaml.dump(combined_dict, f)
            
            self.logger.info(f'Encodings written to {self.__encoding_path} as encodings.yaml')
        
        if inplace:
            self.logger.info("Encoding categoricals inplace")
            self.X_train = enc.transform(self.X_train)
            self.X_test = enc.transform(self.X_test)
            
            self.logger.info("Updating categorical and continuous col lists to reflect encoding")
            self._continuous_cols += self._categorical_cols
            self._categorical_cols = 0

            self.logger.info("Resetting dataframe indices")
            self.X_train.reset_index(drop=True, inplace=True)
            self.X_test.reset_index(drop=True, inplace=True)
        else:
            self.logger.info("Encoding categoricals on copies of X_train, X_test")
            X_train = enc.transform(self.X_train.copy())
            X_test = enc.transform(self.X_test.copy())
            return X_train, X_test

    def winsorize_continuous(self, inplace: bool=True):

        """
        Method that winsorizes continuous features at wins_pctile and
        100-wins_pctile.

        Example: if wins_pctile = 1, then data is winsorized at 1st
        and 99th percentile. 

        If inplace=True, continuous features are modified in place.
        
        Otherwise, method returns a winsorized copy of the continuous
        features in the dataframe.
        """

        self.logger.info(f"Winsorizing continuous columns at {self.__wins_pctile} and {100-self.__wins_pctile} percentiles")
        lower = np.percentile(self.X_train[self._continuous_cols], self.__wins_pctile, axis=0)
        upper = np.percentile(self.X_train[self._continuous_cols], 100-self.__wins_pctile, axis=0)

        if inplace==True:
            self.X_train[self._continuous_cols] = np.clip(self.X_train[self._continuous_cols], lower, upper)
            self.X_test[self._continuous_cols] = np.clip(self.X_test[self._continuous_cols], lower, upper)

        else:
            processed_train = self.X_train[self._continuous_cols].copy()
            processed_train = np.clip(processed_train, lower, upper)

            processed_test = self.X_test[self._continuous_cols].copy()
            processed_test = np.clip(processed_test, lower, upper)
            
            return processed_train, processed_test
        
    def winsorize_label(self, inplace: bool=True):

        """
        Method that winsorizes label at wins_pctile and
        100-wins_pctile.

        Example: if wins_pctile = 1, then data is winsorized at 1st
        and 99th percentile. 

        If inplace=True, label is modified in place.
        
        Otherwise, method returns a winsorized copy of the label.
        """
        self.logger.info(f"Winsorizing label {self._label} at {self.__wins_pctile} and {100-self.__wins_pctile} percentiles")
        lower = np.percentile(self.y_train, self.__wins_pctile, axis=0)
        upper = np.percentile(self.y_train, 100-self.__wins_pctile, axis=0)

        if inplace==True:
            self.y_train = np.clip(self.y_train, lower, upper)
            self.y_test = np.clip(self.y_test, lower, upper)

        else:
            processed_train = self.y_train.copy()
            processed_train= np.clip(processed_train, lower, upper)
            processed_test = self.y_test.copy()
            processed_test= np.clip(processed_test, lower, upper)
            return processed_data

    def _validate_data_for_imputation(self):

        """
        Helper method that validates that self.X_train has no single value columns or 
        mostly null columns before applying impute_missings_with_mice().
        """
        check_cols = [x for x in self.X_train.columns if x not in self._meta_cols]
        if any(self.X_train[col].nunique(dropna=True) <= 1 for col in check_cols):
            self.logger.error("X_train contains single-value columns. Apply drop_single_value_cols() before impute_missings_with_mice().")
            raise ValueError("X_train contains single-value columns. Apply drop_single_value_cols() before impute_missings_with_mice().")
        if any(self.X_train[col].notnull().sum() <= int(np.floor(self.__share_non_null*self.X_train.shape[0])) for col in check_cols):
            self.logger.error("X_train contains mostly null columns. Apply drop_mostly_null_cols() before impute_missings_with_mice().")
            raise ValueError("X_train contains mostly null columns. Apply drop_mostly_null_cols() before impute_missings_with_mice().")
        if self.y_train.isnull().sum() > 0 or self.y_test.isnull().sum() > 0:
            self.logger.error("Label contains null values. Apply drop_null_labels() before impute_missings_with_mice().")
            raise ValueError("Label contains null values. Apply drop_null_labels() before impute_missings_with_mice().")
        
    def impute_missings_with_mice(self, inplace: bool=True):

        """
        Method that imputes missing values in data using miceforest.
        
        Note: this method requires that the user has previously applied drop_single_value_cols(),
        drop_mostly_null_cols(), and drop_null_labels() to prevent downstream errors in miceforest.

        inputs:
            inplace: boolean specifying whether normalization happens in place or
            on a copy of self._data

        If inplace==True, continuous and binary columns of self._data are modified in place. 
        Method does not return anything.

        If inplace==False, method is applied to a copy of self._data.
        Method returns a copy of continuous and binary features in self._data
        whose missing values have been imputed using miceforest.
        """

        # make sure user has run drop_single_value_cols() and drop_mostly_null_cols()
        self._validate_data_for_imputation()

        # reset indices
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        self.meta_train = self.meta_train.reset_index(drop=True)
        self.meta_test = self.meta_test.reset_index(drop=True)
        
        # subset to binary and continuous columns that are still 
        # present in the dataframe after drops from other methods.
        features_to_process = [x for x in self._data.columns if x in self._continuous_cols or x in self._binary_cols]

        train = self.X_train[features_to_process].copy()
        test = self.X_test[features_to_process].copy()

        if all(train[col].isnull().sum() == 0 for col in train.columns):
            self.logger.info("There are no missing values for impute_missings_with_mice() to impute. Skipping imputation.")
            return

        else:
            # Create miceforest kernel
            self.logger.info("Imputing missings with miceforest")
            
            kernel = mf.ImputationKernel(
                        train,
                        num_datasets=1,
                        save_all_iterations_data=True,
                        random_state=self.__random_state
                    )
            
            # Perform imputation on train set using kernel
            kernel.mice(self.__mice_iters)

            #
            print('test/train column info before imputation:')
            print(test.info())
            print(train.info())

            # Extract the imputed train set
            imputed_train = kernel.complete_data(0)

            # Apply fitted imputation to test set
            if any(test[col].isnull().sum() > 0 for col in test.columns):
                imputed_test = kernel.impute_new_data(test).complete_data()
                print(test.info())
                print(train.info())
            else:
                imputed_test = test
                self.logger.info("no missing values in test set")
            self.logger.info("Impute complete")

            # Substitute imputed features into train and test sets
            if inplace==True:
                self.X_train[features_to_process] = imputed_train
                self.X_test[features_to_process] = imputed_test

            else:
                return imputed_train, imputed_test

    def normalize_continuous_cols(self, inplace: bool=True):

        """
        Normalizes continuous features if present in data using sklearn's
        StandardScaler().

        inputs:
            inplace: boolean specifying whether normalization happens in place or
            on a copy of self._data

        If inplace==True, continuous columns of  self._data are modified in place. 
        Method does not return anything.

        If inplace==False, method is applied to a copy of self._data.
        Method returns normalized copy of continuous features in self._data.
        """
        features_to_process = [x for x in self.X_train.columns if x in self._continuous_cols]

        if not features_to_process:
            self.logger.info('Data has no continuous columns; normalize_continuous_cols() not applied.')
            return
            
        self.logger.info("Normalizing continuous columns")
        train_subset = self.X_train[features_to_process].copy()

        # define scaler
        scaler = StandardScaler()

        # fit scaler on train set
        scaler.fit(train_subset)

        if inplace==True:
            # scale X_train, X_test inplace
            self.X_train[features_to_process] = pd.DataFrame(
                scaler.transform(self.X_test[features_to_process]),
                columns=features_to_process,
                index=self.X_train.index
            )

            self.X_test[features_to_process] = pd.DataFrame(
                scaler.transform(self.X_test[features_to_process]),
                columns=features_to_process,
                index=self.X_train.index
            )
        else:
            return scaler.transform(self.X_train[features_to_process]), scaler.transform(self.X_test[features_to_process])
        
    def normalize_binary_cols(self, inplace: bool=True):

        """
        Same as above but for binary columns.

        I've broken the functions out between binary and continuous
        variables because we don't always want or need to normalize
        binary features. 
        """

        features_to_process = [x for x in self._data.columns if x in self._binary_cols]

        if not features_to_process:
            self.logger.info('Data has no binary columns; normalize_binary_cols() not applied.')
            return

        self.logger.info("Normalizing binary columns")
        train_subset = self.X_train[features_to_process].copy()
        
        scaler = StandardScaler()
        scaler.fit(train_subset)

        if inplace==True:
            self.X_train = scaler.transform(self.X_train[features_to_process])
            self.X_test = scaler.transform(self.X_test[features_to_process])
        else:
            return scaler.transform(self.X_train[features_to_process]), scaler.transform(self.X_test[features_to_process])
        
    def run(self, 
            inplace: bool=True,
            one_hot: bool=False,
            target_encode: bool=False,
            normalize_binary: bool=False):

        """
        It's da wrapper

        Note: method as currently written only accommodates inplace modifications.
        Recommended usage: create a copy of whatever data you want to preprocess and 
        set self._data as the copy.

        Inputs:
        - inplace: bool, determines whether modifications are made in place. current version only supports inplace modifications.
        - one_hot: bool. If True, wrapper calls one_hot() to encode categoricals. Default is False.
        - target_encode: bool. If True, wrapper calls target_encode to encode categorical variables.
        - normalize_binary: bool. If True, wrapper calls normalize_binary_cols() to normalize binary columns. Default is False.

        Returns: 
        - self._data, modified inplace.
        """

        if inplace == False:
            self.logger.info("Wrapper run() only supports inplace modifications. Set inplace=True or run methods individually with inplace=False.")
            return

        self.logger.info("Running preprocessing pipeline.")
                
        self.drop_null_labels()
        
        self.drop_single_value_cols()
        
        self.drop_mostly_null_cols()
        
        self.train_test_split()
        
        self._drop_problematic_cols_from_splits()
        
        if one_hot:
            self.one_hot()
        
        if self.__wins_pctile > 0:
            self.winsorize_continuous()
            self.winsorize_label()
        
        if target_encode:
            self.target_encode()
        
        self.impute_missings_with_mice()
        
        self.normalize_continuous_cols()
        
        if normalize_binary:
            self.normalize_binary_cols()
        
        self.logger.info("Preprocessing complete, returning self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test, self._continuous_cols, self._binary_cols, and self._categorical_cols")

        return self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test, self._continuous_cols, self._binary_cols, self._categorical_cols
