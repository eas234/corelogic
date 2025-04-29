import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pickle
import itertools

# Create a directory to store results and intermediate output if not already existing
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Save cross-validation splits as CSV files
def save_fold_splits(X, y, fold_dir, rand_state=42):
    kf = KFold(n_splits=5, shuffle=True, random_state=rand_state)
    fold_num = 1
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Save fold to CSV
        fold_train = pd.concat([X_train, y_train], axis=1)
        fold_test = pd.concat([X_test, y_test], axis=1)
        
        fold_train.to_csv(f'{fold_dir}/train_fold_{fold_num}.csv', index=False)
        fold_test.to_csv(f'{fold_dir}/test_fold_{fold_num}.csv', index=False)
        
        fold_num += 1

# Load previously tested hyperparameters and results from pickle
def load_tested_params(pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    return {}

# Save tested hyperparameters and performance
def save_tested_params(pickle_file, tested_params):
    with open(pickle_file, 'wb') as f:
        pickle.dump(tested_params, f)

# Perform cross-validation and save the performance for each parameter set
def tune_random_forest(X, y, param_grid, fold_dir, pickle_file):
    tested_params = load_tested_params(pickle_file)
    results = []
    
    # Create all possible hyperparameter combinations from the grid
    param_combinations = list(itertools.product(*param_grid.values()))
    
    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        
        # Check if this combination was already tested
        if tuple(param_dict.items()) in tested_params:
            print(f"Skipping {param_dict} as it's already tested.")
            continue
        
        print(f"Testing hyperparameters: {param_dict}")
        
        # Train and evaluate RandomForest with current hyperparameters
        rf = RandomForestClassifier(**param_dict, random_state=42)
        
        # Cross-validation to get the accuracy score
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
        
        # Calculate mean accuracy
        mean_accuracy = accuracies.mean()
        print(f"Mean accuracy for {param_dict}: {mean_accuracy:.4f}")
        
        # Save results
        results.append({'params': param_dict, 'accuracy': mean_accuracy})
        
        # Save tested parameters with results to resume later
        tested_params[tuple(param_dict.items())] = mean_accuracy
        save_tested_params(pickle_file, tested_params)

  def main():
    # Example dataset (replace with your own dataset)
    # X and y should be pandas DataFrame and Series respectively
    X = pd.read_csv('your_data.csv').drop('target', axis=1)  # Replace with your dataset path
    y = pd.read_csv('your_data.csv')['target']  # Replace with your target column name

    # Hyperparameter grid to search over
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Directory to save fold CSVs and other results
    fold_dir = 'folds'
    create_directory(fold_dir)
    
    # Save the folds as CSV files
    save_fold_splits(X, y, fold_dir)

    # Pickle file to save tested hyperparameters and results
    pickle_file = 'tested_hyperparams.pkl'

    # Run the hyperparameter tuning
    results = tune_random_forest(X, y, param_grid, fold_dir, pickle_file)

    # Optionally, print or save the final results
    results_df = pd.DataFrame(results)
    results_df.to_csv('tuning_results.csv', index=False)
    print("Hyperparameter tuning completed. Results saved.")
    
if __name__ == "__main__":
    main()
