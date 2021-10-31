import numpy as np
from costs import compute_mse
from implementations import *
from proj1_helpers import *

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
       
    lambdas = np.logspace(-2,1,40)
    degrees = range(1,10)
    
    min_testerror = np.zeros((len(degrees),len(lambdas)))
    k_indices = build_k_indices(y,k,seed)
    
    for idx_degree,degree in enumerate(degrees):    
        for idx_lambda,lambda_ in enumerate(lambdas):
            mse_test_intermediate = []
            
            for ii in range(k):
                loss_tr_ii,loss_te_ii = cross_validation_degree(y,x,k_indices,ii,lambda_,degree)
                mse_test_intermediate.append(loss_te_ii)

            min_testerror[idx_degree,idx_lambda] = np.mean(mse_test_intermediate)
            
    best_parameters = np.unravel_index(np.argmin(min_testerror, axis=None), min_testerror.shape)
    best_degree = degrees[best_parameters[0]]
    best_lambda = lambdas[best_parameters[1]]
    
    return best_degree, best_lambda


def logistic_cross_validation(y, x, k_indices, k, gamma, lambda_, degree):
    '''calculates the average loss and accuracy for k folded cross-validation '''
    train_indices = np.setdiff1d(k_indices,k_indices[k])
    max_iters_LR = 10000
    gamma_LR = 0.001
    threshold_LR = 1e-3
    
    
    x_test = x[k_indices[k],:]
    x_train = x[train_indices,:]
    y_test = y[k_indices[k]]
    y_train = y[train_indices]
    
    x_train_poly = build_poly(x_train,degree)
    x_test_poly = build_poly(x_test,degree)
    initial_w = np.zeros(x_train_poly.shape[1])
    
    loss_train,w_optimal = reg_logistic_regression(y_train,x_train_poly,lambda_, initial_w, max_iters_LR, gamma_LR, threshold_LR)
    pred = predict_labels(w_optimal,x_test_poly)
    pred_bin = np.squeeze(1*np.equal(pred,1))
    correct = np.sum(1*np.equal(y_test,pred_bin))
    all_ = np.shape(y_test)[0]
    acc =  correct/all_
    print(f"The accuracy is",acc) 

    return loss_train, acc

def logistic_optimisation(y,x,k,gamma,seed=1):
    '''finds the optimal value of lambda by performing cross-validation for lambda values from 1e-4 to 1 '''
    lambdas = np.logspace(-3,1,5)
    degrees = range(1,5)
    
    
    k_indices = build_k_indices(y,k,seed)
    min_testerror = np.zeros((len(degrees),len(lambdas)))
    
    for idx_degree,degree in enumerate(degrees):
        for idx_lambda,lambda_ in enumerate(lambdas):
            accuracy_intermediate = []

            for ii in range(k):
                print(f"Doing fold",ii,"for lambda value:",lambda_,"degree:",degree) 
                loss_tr_ii, acc_test = logistic_cross_validation(y, x, k_indices, ii, gamma, lambda_,degree)
                accuracy_intermediate.append(acc_test)

            min_testerror[idx_degree,idx_lambda] = np.mean(accuracy_intermediate)
    
    best_parameters = np.unravel_index(np.argmin(min_testerror, axis=None), min_testerror.shape)
    best_degree = degrees[best_parameters[0]]
    best_lambda = lambdas[best_parameters[1]]
    
    return best_degree, best_lambda