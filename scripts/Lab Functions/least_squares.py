# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares."""
    # Solving normal equaions X.T (y - Xw) = 0
    gram = tx.T @ tx
    w = np.linalg.solve(gram, tx.T@y)
    
    # Calculate MSE
    mse = compute_mse(y, tx, w)
    
    return mse, w
