# -*- coding: utf-8 -*-
"""This file reproduces the results for the best predictions submitted in 
   https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/submissions
   ID: 164279"""

import numpy as np
from proj1_helpers import *
from implementations import *
from data_processing import *

""" Download training data """
DATA_TRAIN_PATH = 'train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

""" Processing Training data """
thresh = 0
std_limit = 5
processed_data, most_frq_mass, jet_num_idx, possible_jets, idx_out, all_mean, all_std = process_data(tX, thresh, std_limit)

""" Best hyperparametres found """
degrees = [3, 3, 3, 3]
lambdas = [0.05878016072274915, 0.49238826317067413, 1.0, 0.001]

# Initialize weights
w_RR = []

""" For each group run Ridge Regression """
for jet in possible_jets:
    y_data = y[jet_num_idx[jet][0]]
        
    # Build polynomial
    tx = build_poly(processed_data[jet], degrees[jet])
    
    # Ridge Regression 
    loss, w = ridge_regression(y_data, tx, lambdas[jet])
    w_RR.append(w)



""" Download testing data """
DATA_TEST_PATH = 'test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

""" Processing Test data """
jet_num_idx = grouping(tX_test)
tx_imp = tX_test
tx_imp[:, 0] = np.where(tx_imp[:, 0] == -999, most_frq_mass, tx_imp[:, 0])
tx_imp[jet_num_idx[0], :] = np.where(tx_imp[jet_num_idx[0], :] == -999, 0, tx_imp[jet_num_idx[0], :])
tx_imp[jet_num_idx[1], :] = np.where(tx_imp[jet_num_idx[1], :] == -999, 0, tx_imp[jet_num_idx[1], :])

""" Create predictions """
OUTPUT_PATH = 'predictions.csv'

processed_tX_test= []
y_pred = np.empty(tX_test.shape[0])

for jet in possible_jets:
    data = tx_imp[jet_num_idx[jet][0], :]
    num_samples = data.shape[0]
    all_idx = range(data.shape[1])    
    processed_tX_test.append(data[:, np.setdiff1d(all_idx, idx_out[jet])])
    
    # Handle outliers and standardize
    processed_tX_test[jet] = clip_outliers(processed_tX_test[jet], std_limit, all_mean[jet], all_std[jet])
    test_offset = build_poly(processed_tX_test[jet], degrees[jet])
    
    # Create preditions for each jet number
    y_pred[jet_num_idx[jet][0]] = predict_labels(w_optimal[jet], test_offset)

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
