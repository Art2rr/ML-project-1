import numpy as np
import time
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

def build_poly(x, degree):
    phi = np.zeros([x.shape[0], (degree+1)*x.shape[1]])
    idx = 0
    for j in range(degree+1):
        for k in range(x.shape[1]):
            phi[:,idx] = x[:,k]**j
            idx += 1
    return phi

def cross_validation_degree(y,x,k_indices,k,lambda_,degree):
   
    train_indices = np.setdiff1d(k_indices,k_indices[k])
    
    x_test = x[k_indices[k],:]
    x_train = x[train_indices,:]
    y_test = y[k_indices[k]]
    y_train = y[train_indices]
    
    x_train_poly = build_poly(x_train,degree)
    x_test_poly = build_poly(x_test,degree)

    loss_train,w_optimal = ridge_regression(y_train,x_train_poly,lambda_)
    loss_test = compute_mse(y_test,x_test_poly,w_optimal)

    return loss_train,loss_test

def find_hyperparameters(y,x,k,seed=1):
    
    current_time = time.time()
    
    lambdas = np.logspace(-4,0,10)
    degrees = range(1,5)
    
    min_testerror = np.zeros((len(degrees),len(lambdas)))
    k_indices = build_k_indices(y,k,seed)
    
    for idx_degree,degree in enumerate(degrees):    
        for idx_lambda,lambda_ in enumerate(lambdas):
            mse_test_intermediate = []
            print(degree,lambda_)
    
            for ii in range(k):
                loss_tr_ii,loss_te_ii = cross_validation_degree(y,x,k_indices,ii,lambda_,degree)
                mse_test_intermediate.append(loss_te_ii)

            min_testerror[idx_degree,idx_lambda] = np.mean(mse_test_intermediate)
        
        print(time.time()-current_time)
    
    print(min_testerror)
    best_parameters = np.unravel_index(np.argmin(min_testerror, axis=None), min_testerror.shape)
    print(best_parameters)
    best_degree = degrees[best_parameters[0]]
    best_lambda = lambdas[best_parameters[1]]
    
    return best_degree, best_lambda
