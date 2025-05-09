import datetime
import logging
import miceforest as mf
import numpy as np
import os
import pandas as pd

class Preprocess:
    """
    Class which contains methods to preprocess data.

    Methods include:
    
    - Setters and getters for all class attributes
    - drop_null_labels(): drops rows where label is null
    - drop_single_value_cols(): drops columns that have only one value
    - drop_mostly_null_cols(): drops columns that have fewer than n_non_null non-null values
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
                 n_non_null: int=0, # minimum number of non-null values required in each column
                 random_state: int=42, # for reproducibility
                 wins_pctile: int=1, # percentile at which data are winsorized (symmetric)
                 mice_iters: int=3, # n_iters for miceforest imputer
                 log_dir: str='logs' # logger filepath
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
        
        # public attributes
        self._data = data
        self._label = label or ''
        self._continuous_cols = continuous_cols or []
        self._binary_cols = binary_cols or []
        self._categorical_cols = categorical_cols or []
        self._meta_cols = meta_cols or []

        # private attributes
        self.__n_non_null = n_non_null
        self.__random_state = random_state
        self.__wins_pctile = wins_pctile
        self.__mice_iters = mice_iters

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
    def labels(self, new_label):
        if isinstance(new_label, str):
            self._labels = label
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

    # min non-null values
    @property
    def n_non_null(self):
        return self.__n_non_null
    
    @n_non_null.setter
    def n_non_null(self, new_n):
        if new_n >= 0 and isinstance(new_n, int):
            self.__n_non_null = new_n
        else:
            self.logger.error("n_non_null must be a positive integer")
            raise ValueError("n_non_null must be a positive integer")

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
        drop_cols = [col for col in self._data.columns if self._data[col].nunique(dropna=True) <= 1]

        # drop single-value columns if found
        if drop_cols:
            self.logger.warning(f"Warning: {len(drop_cols)} columns have only one value. Dropping {drop_cols}")
        else:
            self.logger.info("No single-value columns to drop.")
            pass

        if inplace==True:
            self._data.drop(columns=drop_cols, inplace=True)
            self._data.reset_index(drop=True, inplace=True)
            self.logger.info("Updating cols attributes to reflect drops")
            for attr in [self._continuous_cols, self._binary_cols, self._categorical_cols]:
                attr = [x for x in attr if x in self.data.columns]
        else:
            processed_data = self._data.copy()
            processed_data = processed_data.drop(columns=drop_cols)
            processed_data = processed_data.reset_index(drop=True)
            return processed_data

    def drop_mostly_null_cols(self, inplace: bool=True):

        """
        Drop columns in input data that have fewer than n_non_null 
        non-null values. 

        If inplace==True, columns are dropped in place and self._data indices are reset after drop.
        Method does not return anything.

        Else, method returns copy of self._data with dropped columns.
        """

        # identify mostly null cols
        drop_cols = [col for col in self._data.columns if self._data[col].notnull().sum() <= self.__n_non_null]

        if drop_cols:
            self.logger.warning(f'Warning: {len(drop_cols)} columns have fewer than {self.__n_non_null} non-null values. Dropping {drop_cols}')
        else:
            self.logger.info('No mostly null columns to drop.')
            pass

        if inplace==True:
            self._data.drop(columns=drop_cols, inplace=True)
            self._data.reset_index(drop=True, inplace=True)
            self.logger.info("Updating cols attributes to reflect drops")
            for attr in [self._continuous_cols, self._binary_cols, self._categorical_cols]:
                attr = [x for x in attr if x in self.data.columns]

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
        lower = np.percentile(self._data[self._continuous_cols], self.__wins_pctile, axis=0)
        upper = np.percentile(self._data[self._continuous_cols], 100-self.__wins_pctile, axis=0)

        if inplace==True:
            self._data[self._continuous_cols] = np.clip(self._data[self._continuous_cols], lower, upper)

        else:
            processed_data = self._data[self._continuous_cols].copy()
            processed_data = np.clip(processed_data, lower, upper)
            return processed_data
        
    def winsorize_label(self, inplace: bool=True):

        """
        Method that winsorizes label at wins_pctile and
        100-wins_pctile.

        Example: if wins_pctile = 1, then data is winsorized at 1st
        and 99th percentile. 

        If inplace=True, label is modified in place.
        
        Otherwise, method returns a winsorized copy of the label.
        """
        self.logger.info(f"Winsorizing label at {self.__wins_pctile} and {100-self.__wins_pctile} percentiles")
        lower = np.percentile(self._data[self._label], self.__wins_pctile, axis=0)
        upper = np.percentile(self._data[self._label], 100-self.__wins_pctile, axis=0)

        if inplace==True:
            self._data[self._label] = np.clip(self._data[self._label], lower, upper)

        else:
            processed_data = self._data[self._label].copy()
            processed_data = np.clip(processed_data, lower, upper)
            return processed_data

    def _validate_data_for_imputation(self):

        """
        Helper method that validates that self._data has no single value columns or 
        mostly null columns before applying impute_missings_with_mice().
        """

        if any(self._data[col].nunique(dropna=True) <= 1 for col in self._data.columns):
            self.logger.error("Data contains single-value columns. Apply drop_single_value_cols() before impute_missings_with_mice().")
            raise ValueError("Data contains single-value columns. Apply drop_single_value_cols() before impute_missings_with_mice().")
        if any(self._data[col].notnull().sum() <= self._n_non_null for col in self._data.columns):
            self.logger.error("Data contains mostly null columns. Apply drop_mostly_null_cols() before impute_missings_with_mice().")
            raise ValueError("Data contains mostly null columns. Apply drop_mostly_null_cols() before impute_missings_with_mice().")
        if self._data[self._label].isnull().sum() > 0:
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

        # subset to binary and continuous columns that are still 
        # present in the dataframe after drops from other methods.
        features_to_process = [x for x in processed_data.columns if x in self._continuous_cols or x in self._binary_cols]

        subset = self._data[features_to_process].copy()
        if all(subset[col].isnull().sum() == 0 for col in subset.columns):
            self.logger.info("There are no missing values for impute_missings_with_mice() to impute. Skipping imputation.")
            return

        else:
            # Create miceforest kernel
            kernel = mf.ImputationKernel(
                        subset,
                        num_datasets=1,
                        save_all_iterations_data=False,
                        random_state=self.__random_state
                    )
            
            # Perform imputation using kernel
            kernel.mice(self.__mice_iters)

            # Extract the imputed dataset
            imputed_subset = kernel.complete_data(0)

            # Substitute imputed features into data
            if inplace==True:
                self._data[features_to_process] = imputed_subset
            else:
                return imputed_subset

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
        features_to_process = [x for x in self._data.columns if x in self._continuous_cols]

        if not features_to_process:
            self.logger.info('Data has no continuous columns; normalize() not applied.')
            return
        
        subset = self._data[features_to_process].copy()
        
        scaler = StandardScaler()
        normalized = scaler.fit_transform(subset)

        if inplace==True:
            self._data[features_to_process] = normalized
        else:
            return normalized
        
    def normalize_binary_cols(self, inplace: bool=True):

        """
        Same as above but for binary columns.

        I've broken the functions out between binary and continuous
        variables because we don't always want or need to normalize
        binary features. 
        """

        features_to_process = [x for x in self._data.columns if x in self._binary_cols]

        if not features_to_process:
            self.logger.info('Data has no binary columns; normalize() not applied.')
            return
        
        subset = self._data[features_to_process].copy()
        
        scaler = StandardScaler()
        normalized = scaler.fit_transform(subset)

        if inplace==True:
            self._data[features_to_process] = normalized
        else:
            return normalized
        
    def run(self, 
            inplace: bool=True,
            one_hot: bool=False,
            normalize_binary: bool=False):

        """
        It's da wrapper

        Note: method as currently written only accommodates inplace modifications.

        Inputs:
        - inplace: bool, 
        """

        if inplace == False:
            self.logger.info("Wrapper run() only supports inplace modifications. Set inplace=True or run methods individually with inplace=False.")
            return

        self.logger.info("Running full preprocessing pipeline.")
        self.drop_null_labels()
        self.drop_single_value_cols()
        self.drop_mostly_null_cols()
        if one_hot:
            self.one_hot()
        if self.__wins_pctile > 0:
            self.winsorize_continuous()
            self.winsorize_label()
        self.impute_missings_with_mice()
        self.normalize_continuous_cols()
        if normalize_binary:
            self.normalize_binary_cols()

        return self._data
