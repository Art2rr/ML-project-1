import numpy as np

def sigmoid(t):
    '''sigmoid function for a column vector. The vlaue is bounded from below at -100 to prevent overflow'''
    z = np.squeeze(t)
    z_bound = np.where(z<-100,-100,1)
    sigma = 1.0 / (1+np.exp(-z_bound))
    sigma = np.expand_dims(sigma, axis=1)
    return sigma


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sigma = sigmoid(tx.dot(w))
    loss = (y.T.dot(np.log(sigmoid(tx.dot(w)))) + (1-y).T.dot(np.log(1-sigmoid(tx.dot(w)))))
    return np.squeeze(-loss)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigma = sigmoid(tx.dot(w))
    ty = np.expand_dims(y,axis=1)
    grad = tx.T.dot((np.subtract(sigma, ty)))
    return grad


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient for logistic regression. loss value has a penalty dependent of lambda"""
    penalty = np.squeeze(lambda_*w.T.dot(w))
    loss = calculate_loss(y,tx,w) + penalty 
    grad = calculate_gradient(y,tx,w) + 2*lambda_*w
    return loss, grad

