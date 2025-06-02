# Corelogic
Repository for corelogic projects with Stanford Reglab.

# Repository Contents
## Code
* census.py: utilities for processing and merging census tract and block-group level data. also includes utility for reverse geocoding sales that have incorrect or missing FIPS/Census IDs.
* preprocess.py: class that handles data preprocessing pipeline. wrapper run() implements the pipepline.
* modeling_utils.py: utilities for modeling pipeline. implements optuna for model tuning and currently accommodates lasso, RF, and lightGBM as objectives. tune_model() tunes the model and rf_train_test_write() trains, tests, and writes the output. need to update the latter to accommodate lasso and lightGBM
* evaluation.py: module with functions to evaluate model outputs. Currently contains a couple plotting utilities (e.g. binnedDotPlot()) and evaluation metrics (e.g. cod(), prb())
### Models
* run_model.py: script to run model, governed by config
### Old
deprecated code.
## Config
Directory to store config files for each model. 
