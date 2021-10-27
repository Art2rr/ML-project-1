# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse.
    """
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))
    
    return loss
    
    
def compute_mae(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mae.
    """
    N = len(y)
    error = np.abs(y - (tx @ w))
    loss = np.sum(error)/(N)
    
    return loss

# # -*- coding: utf-8 -*-
# """A function to compute the cost."""


# def compute_mse(y, tx, w):
#     """compute the loss by mse."""
#     e = y - tx.dot(w)
#     mse = e.dot(e) / (2 * len(e))
#     return mse