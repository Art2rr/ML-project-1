# -*- coding: utf-8 -*-
"""Function implementations for project 1."""
import numpy as np
from costs import compute_mse
from helper_GD_SGD import compute_gradient, batch_iter
from helpers_LR import *


def least_squares(y, tx):
    """calculate the least squares."""
    # Solving normal equations X.T (y - Xw) = 0
    gram = tx.T @ tx
    w = np.linalg.solve(gram, tx.T@y)
    
    # Calculate MSE
    loss = compute_mse(y, tx, w)    
    return loss, w


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad
    loss = compute_mse(y, tx, w)
    return loss, w


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma*grad
    loss = compute_mse(y, tx, w)
    return loss, ws


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""    
    gram = tx.T @ tx
    N = len(gram)
    lambda_lin = lambda_ * 2*N
    w_ridge = np.linalg.inv(gram + (lambda_lin*np.eye(N))) @ tx.T @ y
    loss = compute_mse(y, tx, w_ridge)
    return loss, w_ridge


def logistic_regression(y, tx, initial_w, max_iter, gamma, threshold):
    '''implementation of logistic regression using full gradient descent. The interations run until the convergence threshold. To force the convergence the step size is halved every 10 iterations'''
    #initialise w
    w = np.expand_dims(initial_w,axis=1)
    losses = []

    #logistic regression
    for iter in range(max_iter):
        
        # get loss and update w.
        loss, grad = penalized_logistic_regression(y, tx, w, 0)
        w = w - (gamma * grad)
        
        # decrease step size
        if iter % 10 == 0:
            gamma = gamma*0.5
            
        # log info -- PRINTIG DISABLED
        #if iter % 100 == 0:
        #    print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    l = calculate_logistic_loss(y, tx, w)
    #print("Final loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return l, w

    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma, threshold):
    '''implementation of penalised logistic regression using full gradient descent. Parameter lambda regularises the loss calculation. The interations run until the convergence threshold. To force the convergence the step size is halved every 10 iterations'''
    #initialise w
    w = np.expand_dims(initial_w,axis=1)
    losses = []

    #logistic regression
    for iter in range(max_iter):
        
        # get loss and update w.
        loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - (gamma * grad)
        
        # decrease step size
        if iter % 10 == 0:
            gamma = gamma*0.5
            
        # log info -- PRINTIG DISABLED
        #if iter % 100 == 0:
        #    print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    l = calculate_logistic_loss(y, tx, w)
    #print("Final loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return l, w
