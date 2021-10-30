import numpy as np
from proj1_helpers import *
from implementations import *
from data_processing1 import *
from lambda_cross_validation import *
from build_polynomial import *
from hyperparameters import *

# Download training data
DATA_TRAIN_PATH = 'train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Training data processing
thresh = 0
std_limit = 5
processed_data, jet_num_idx, possible_jets, idx_out, all_mean, all_std = process_data(tX, thresh, std_limit)

#------------------------------------

max_iters_GD = 10
gamma = 0.1

max_iters_LR = 10000
threshold_LR = 1e-8
gamma_LR = 0.01

k_fold = 4
degrees = []
lambdas = []

# loss_GD = []
# w_GD = []

# loss_SGD = []
# w_SGD = []

# loss_LS = []
# w_LS = []

loss_RR = []
w_RR = []

loss_LR = []
w_LR = []

loss_RLR = []
w_RLR = []

for jet in possible_jets:
    y_data = y[jet_num_idx[jet][0]]
    initial_w = np.zeros(processed_data[jet].shape[1]+1)
    
    num_samples = len(y_data)
    tx_offset = np.c_[np.ones(num_samples), processed_data[jet]]
    
#     # Gradient descent
#     loss, w = least_squares_GD(y_data, tx_offset, initial_w, max_iters_GD, gamma)
#     loss_GD.append(loss)
#     w_GD.append(w)
    
#     # Stochastic gradient descent
#     loss, w = least_squares_SGD(y_data, tx_offset, initial_w, max_iters_GD, gamma)
#     loss_SGD.append(loss)
#     w_SGD.append(w)
    
#     # Least Squares
#     loss, w = least_squares(y_data, tx_offset)
#     loss_LS.append(loss)
#     w_LS.append(w)
    
    # Ridge Regression        
    best_degree, best_lambda = find_hyperparameters(y_data, processed_data[jet], k_fold,seed=1)
    degrees.append(best_degree)
    lambdas.append(best_lambda)
    
    tx = build_poly(processed_data[jet], degrees[jet])
    
    loss, w = ridge_regression(y_data, tx, lambdas[jet])
    loss_RR.append(loss)
    w_RR.append(w)
    
    initial_w = np.zeros(tx.shape[1])
    
    # Logistic Regression
    loss, w = logistic_regression(y_data, tx, initial_w, max_iters_LR, gamma_LR, threshold_LR)
    loss_LR.append(loss)
    w_LR.append(w)
    
    # Regularized Logistic Regression
    loss, w = reg_logistic_regression(y_data, tx, lambdas[jet], initial_w, max_iters_LR, gamma_LR)
    loss_RLR.append(loss)
    w_RLR.append(w)
#------------------------------------



#Download testing data
DATA_TEST_PATH = 'test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Testing data processing
jet_num_idx = grouping(tX_test)
tx_imp_test = imputation (tX_test, jet_num_idx)

# Create predictions
OUTPUT_PATH = 'predictions.csv'

processed_tX_test= []
y_pred = np.empty(tX_test.shape[0])

for jet in possible_jets:
    data = tx_imp_test[jet_num_idx[jet][0], :]
    num_samples = data.shape[0]
    all_idx = range(data.shape[1])
    
    processed_tX_test.append(data[:, np.setdiff1d(all_idx, idx_out[jet])])
    processed_tX_test[jet] = clip_outliers(processed_tX_test[jet], std_limit, all_mean[jet], all_std[jet])
    # test_offset = np.c_[np.ones(num_samples), processed_tX_test[jet]]
    test_offset = build_poly(processed_tX_test[jet], degree=3)
    
    y_pred[jet_num_idx[jet][0]] = predict_labels(w_RR_4[jet], test_offset)

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)