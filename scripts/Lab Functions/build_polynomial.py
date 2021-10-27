# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.zeros([len(x), degree+1])
    for i in range(degree+1):
        phi[:,i] = x**i
    return phi