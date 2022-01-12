"""
Based on example in ../notebooks/credit.ipynb
"""

import click
import os
import time

import sage
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier


@click.command()
@click.option("--model-filename", default="credit_model.cbm")
def main(model_filename):
    categorical_inds, feature_names, train, val, test, Y_train, Y_val, Y_test = load_data()

    if os.path.isfile(model_filename):
        model = load_model(model_filename)
    else:
        model = train_model(categorical_inds, train, val, Y_train, Y_val)
        model.save_model(model_filename)

    evaluate_model(test, Y_train, Y_test, model)

    test_with_cross_entropy(model, train, test, Y_test)

    test_with_unfairness_metric(model, train, test, Y_test, feature_names)

############################
### Section 1: Load data ###
############################

def load_data():
    """Load German credit data and split into train, val, and test sets."""
    # Load data
    df = sage.datasets.credit()

    # Feature names and categorical columns (for CatBoost model)
    feature_names = df.columns.tolist()[:-1]
    categorical_columns = [
        'Checking Status', 'Credit History', 'Purpose', 'Credit Amount',
        'Savings Account/Bonds', 'Employment Since', 'Personal Status',
        'Debtors/Guarantors', 'Property Type', 'Other Installment Plans',
        'Housing Ownership', 'Job', 'Telephone', 'Foreign Worker'
    ]
    categorical_inds = [feature_names.index(col) for col in categorical_columns]

    # Split data
    train, test = train_test_split(
        df.values, test_size=int(0.1 * len(df.values)), random_state=0)
    train, val = train_test_split(
        train, test_size=int(0.1 * len(df.values)), random_state=0)
    Y_train = train[:, -1].copy().astype(int)
    Y_val = val[:, -1].copy().astype(int)
    Y_test = test[:, -1].copy().astype(int)
    train = train[:, :-1].copy()
    val = val[:, :-1].copy()
    test = test[:, :-1].copy()

    return categorical_inds, feature_names, train, val, test, Y_train, Y_val, Y_test

##############################
### Section 2: Train model ###
##############################

def load_model(model_filename):
    model = CatBoostClassifier()
    model.load_model(model_filename)
    return model


def train_model(categorical_inds, train, val, Y_train, Y_val):
    model = CatBoostClassifier(iterations=50,
                            learning_rate=0.3,
                            depth=3)

    model = model.fit(train, Y_train, categorical_inds, eval_set=(val, Y_val),
                      verbose=False)

    return model


def evaluate_model(test, Y_train, Y_test, model):
    # Calculate performance
    p = np.array([np.sum(Y_train == i) for i in np.unique(Y_train)]) / len(Y_train)
    base_ce = log_loss(Y_test.astype(int), p[np.newaxis].repeat(len(test), 0))
    ce = log_loss(Y_test.astype(int), model.predict_proba(test))

    print('Base rate cross entropy = {:.3f}'.format(base_ce))
    print('Model cross entropy = {:.3f}'.format(ce))

###############################################################
### Section 3: Calculate importance (marginal distribution) ###
###############################################################

def test_with_cross_entropy(model, train, test, Y_test):
    """ORIGINAL: Cross entropy as loss function"""

    # Setup and calculate
    imputer = sage.MarginalImputer(model, train[:512])
    estimator = sage.PermutationEstimator(imputer, 'cross entropy')
    tic = time.perf_counter()
    sage_values = estimator(test, Y_test)
    toc = time.perf_counter()

    # Print results
    print("SAGE values using cross entropy as loss:", sage_values)
    print(f"Calculated in {(toc - tic)/60:0.4f} minutes")


def test_with_unfairness_metric(model, train, test, Y_test, feature_names):
    """NEW: Equal error rate as loss function"""

    # Setup and calculate with custom fairness-related loss function
    imputer = sage.MarginalImputer(model, train[:512])
    estimator_eer = sage.PermutationEstimator(imputer, 'equal error rate')
    sensitive_column_index = feature_names.index("Age")
    check_in_sensitive_group = lambda data: data[:,sensitive_column_index] <= 30
    tic_eer = time.perf_counter()
    sage_values_eer = estimator_eer(test, Y_test, check_in_sensitive_group=check_in_sensitive_group)
    toc_eer = time.perf_counter()

    # Print results
    print("SAGE values using equal error rate as loss:", sage_values_eer)
    print(f"Calculated in {(toc_eer - tic_eer)/60:0.4f} minutes")


if __name__ == '__main__':
    main()
