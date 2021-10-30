# -*- coding: utf-8 -*-
"""Function implementations for project 1."""
import numpy as np
from costs import compute_mse
from helper_GD_SGD import compute_gradient, batch_iter


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
            # grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma*grad
    loss = compute_mse(y, tx, w)
    return loss, w


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""    
    gram = tx.T @ tx
    N = len(gram)
    lambda_lin = lambda_ * 2*N
    w_ridge = np.linalg.inv(gram + (lambda_lin*np.eye(N))) @ tx.T @ y
    loss = compute_mse(y, tx, w_ridge)
    return loss, w_ridge


def sigmoid(t):
    '''sigmoid function for a column vector. The vlaue is bounded from below at -100 to prevent overflow.'''
    z = np.squeeze(t)
    z_bound = np.where(z<-100,-100,1)
    sigma = 1.0 / (1+np.exp(-z_bound))
    sigma = np.expand_dims(sigma, axis=1)
    return sigma


def calculate_logistic_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sigma = sigmoid(tx.dot(w))
    loss = (y.T.dot(np.log(sigmoid(tx.dot(w)))) + (1-y).T.dot(np.log(1-sigmoid(tx.dot(w)))))
    return np.squeeze(-loss)


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigma = sigmoid(tx.dot(w))
    ty = np.expand_dims(y,axis=1)
    grad = tx.T.dot((np.subtract(sigma, ty)))
    return grad


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient for logistic regression. loss value has a penalty dependent of lambda"""
    penalty = np.squeeze(lambda_*w.T.dot(w))
    loss = calculate_logistic_loss(y,tx,w) + penalty 
    grad = calculate_logistic_gradient(y,tx,w)
    return loss, grad


def logistic_regression_GD(y, tx, gamma, lambda_, max_iter=10000):
    '''performs gradient descent for the logistic regression.'''
    #initialise w
    w = np.zeros((tx.shape[1], 1))
    losses = []

    #logistic regression
    for iter in range(max_iter):
        
        # get loss and update w.
        loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - (gamma * grad)
        
        # decrease step size
        if iter % 10 == 0:
            gamma = gamma*0.5
            
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    l = calculate_logistic_loss(y, tx, w)
    print("Final loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return l, w


def logistic_regression_SGD(y, tx, gamma, lambda_, max_iter=10000, batch_size=1):
    '''performs stochastic gradient descent for the logistic regression. Can also be used for batch GD if the batch_size is bigger than 1.'''
    #initialise w
    w = np.zeros((tx.shape[1], 1))
    losses = []

    #logistic regression
    for iter in range(max_iter):
        
        # get loss and update w.
        for y_b, tx_b in batch_iter(y, tx, batch_size):
            loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
            w = w - (gamma * grad)
        
        # decrease step size
        if iter % 10 == 0:
            gamma = gamma*0.5
            
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    l = calculate_logistic_loss(y, tx, w)
    print("Final loss={l}".format(l=calculate_logistic_loss(y, tx, w)))
    return l, w