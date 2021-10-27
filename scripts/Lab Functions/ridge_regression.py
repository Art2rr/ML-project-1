# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    gram = tx.T @ tx
    N = len(gram)
    lambda_lin = lambda_ * 2*N
    w_ridge = np.linalg.inv(gram + (lambda_lin*np.eye(N))) @ tx.T @ y

    return w_ridge
