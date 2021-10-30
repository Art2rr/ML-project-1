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
    losses = []    
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, initial_w, 0, gamma)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break    
    return loss, w

    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma, threshold):
    losses = []
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, initial_w, lambda_, gamma)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break    
    return loss, w
