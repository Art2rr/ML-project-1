# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    i_train = np.ceil(ratio*len(x))
    
    # x
    x_train = np.random.choice(x, i_train.astype(int), replace=False)
    x_test = np.setdiff1d(x, x_train)
    x_index_tr = (x[:, None] == x_train).argmax(axis=0)
    x_index_te = (x[:, None] == x_test).argmax(axis=0)
    
    # y
    y_train = y[x_index_tr]
    y_test = y[x_index_te]
    
    return x_train, x_test, y_train, y_test
