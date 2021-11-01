# -*- coding: utf-8 -*-
"""Data processing functions for project 1."""

import numpy as np

def standardize(x, mean_x, std_x):
    """Standardize the original data set."""
    x = x - mean_x
    x = x / std_x
    return x


def grouping(tX):
    """ Grouping data according to jet number """
    jet_num_idx = []
    jet_num_idx.append(np.where(tX[:,22] == 0))
    jet_num_idx.append(np.where(tX[:,22] == 1))
    jet_num_idx.append(np.where(tX[:,22] == 2))
    jet_num_idx.append(np.where(tX[:,22] == 3))

    return jet_num_idx


def imputation(data, jet_num_idx):
    """ Imputation non-defined values """
    
    # Imputation of the mass column with the most frequent value
    tx_imp = data.copy()
    good_idx = np.where(data[:, 0] != -999)
    round_values = np.round(data[good_idx, 0]).astype(int)
    counts = np.bincount(round_values[0,:])
    tx_imp[:, 0] = np.where(tx_imp[:, 0] == -999, np.argmax(counts), tx_imp[:, 0]) 
    most_frq = np.argmax(counts)

    # Imputation of data for jet_num = 0 and jet_num = 1
    tx_imp[jet_num_idx[0], :] = np.where(tx_imp[jet_num_idx[0], :] == -999, 0, tx_imp[jet_num_idx[0], :])
    tx_imp[jet_num_idx[1], :] = np.where(tx_imp[jet_num_idx[1], :] == -999, 0, tx_imp[jet_num_idx[1], :])

    return tx_imp, most_frq


def variance(data, thresh):
    """ Calculate variance for each feature and identify var = 1 """
    v_vector = np.var(data, axis=0)
    index_out = np.where(v_vector <= thresh)
    return index_out[0]


def clip_outliers(data, std_limit, mean_x, std_x):
    """ Clip outliers bigger than a certain standard deviation limit (std_limit)"""
    num_datapoints = np.shape(data)[0]
    num_feat = np.shape(data)[1]
    indices = np.indices((1,num_datapoints))
    
    # Standardize each feature
    standardized = standardize(data, mean_x, std_x)
    number_outliers = np.zeros((1,num_feat))
    index_outliers = []

    # Find outliers and clip them
    for ii in range(num_feat):
        pos_outlier = standardized[:,ii]>std_limit
        neg_outlier = standardized[:,ii]<-std_limit
        
        for jj in range(num_datapoints):
            if pos_outlier[jj] == True:
                standardized[jj,ii] = std_limit
            if neg_outlier[jj] == True:
                standardized[jj,ii] = -std_limit
    
    return standardized

    
def process_data(data, thresh, std_limit):#, thresh_corr):
    """Processes training data."""
    # Create groups according to jet number
    jet_num_idx = grouping(data)
    
    # Imputation
    imp_data, most_frq = imputation (data, jet_num_idx)    
    possible_jets = range(len(jet_num_idx))
    
    processed_data = []
    all_mean = []
    all_std = []
    idx_out = []
    
    # For each group remove columns with zero variance and clip outliers
    for jet in possible_jets:
        new_data = imp_data[jet_num_idx[jet][0], :]
        
        idx_out.append(variance(new_data, thresh))
        all_idx = range(new_data.shape[1])
        new_data1 = new_data[:, np.setdiff1d(all_idx, idx_out[jet])]
        
        all_mean.append(np.mean(new_data1,0))
        all_std.append(np.std(new_data1,0))
        data_std = clip_outliers(new_data1, std_limit, all_mean[jet], all_std[jet])
        
        processed_data.append(data_std)
    
    # The values for the most frequent mass value, mean values of the training data and standard deviation for the training data
    # are returned to be used in the pre-processing of the test data
    return processed_data, most_frq, jet_num_idx, possible_jets, idx_out, all_mean, all_std