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
