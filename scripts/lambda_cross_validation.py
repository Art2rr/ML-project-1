import numpy as np
from costs import compute_mse
from implementations import *

def build_k_indices(y,k_fold,seed):
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    return np.array(k_indices)

def cross_validation(y,x,k_indices,k,lambda_):
   
    train_indices = np.setdiff1d(k_indices,k_indices[k])
    
    x_test = x[k_indices[k],:]
    x_train = x[train_indices,:]
    y_test = y[k_indices[k]]
    y_train = y[train_indices]

    loss_train,w_optimal = ridge_regression(y_train,x_train,lambda_)
    loss_test = compute_mse(y_test,x_test,w_optimal)

    return loss_train,loss_test

def lambda_optimisation(y,x,k,seed=1):
    
    lambdas = np.logspace(-4, 0, 30)
    k_indices = build_k_indices(y,k,seed)
    mse_test = []
    
    for lambda_ in lambdas:
        mse_test_intermediate = []
    
        for ii in range(k):
            loss_tr_ii,loss_te_ii = cross_validation(y, x, k_indices, ii, lambda_)
            mse_test_intermediate.append(loss_te_ii)
        
        mse_test.append(np.mean(mse_test_intermediate))
    
    idx_min_error = np.argmin(mse_test)
    lambda_optimal = lambdas[idx_min_error]
    
    return lambda_optimal
    