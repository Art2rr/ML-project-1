import numpy as np

def sigmoid(t):
    z =np.where(t >= 0, 1 / (1 + np.exp(-t)), np.exp(t) / (1 + np.exp(t)))
    return z / (1 + z)


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sigma = sigmoid(tx.dot(w))
    loss = (y.T.dot(np.log(sigma)) + (1-y).T.dot(np.log(1-sigma)))
    return np.squeeze(-loss)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigma = sigmoid(tx.dot(w))
    return tx.T.dot(sigma - y)


def learning_by_penalized_gradient(y, tx, w, lambda_, gamma):
    """return the loss, w"""
    loss = calculate_loss(y,tx,w) + lambda_*w.T.dot(w)
    grad = calculate_gradient(y,tx,w) + 2*lambda_*w
    w = w - (gamma * grad)
    return loss, w

