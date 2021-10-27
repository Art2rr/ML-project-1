# -*- coding: utf-8 -*-
"""Data processing functions for project 1."""

import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,0)
    x = x - mean_x
    std_x = np.std(x,0)
    x = x / std_x
    return x


# Grouping according to jet number
def grouping(tX):
    jet_num_idx = []
    jet_num_idx.append(np.where(tX[:,22] == 0))
    jet_num_idx.append(np.where(tX[:,22] == 1))
    jet_num_idx.append(np.where(tX[:,22] == 2))
    jet_num_idx.append(np.where(tX[:,22] == 3))

    return jet_num_idx



def imputation (data, jet_num_idx):
    # Imputation the mass column with the most frequent value
    tx_imp = data.copy()
    good_idx = np.where(data[:, 0] != -999)
    round_values = np.round(data[good_idx, 0]).astype(int)
    counts = np.bincount(round_values[0,:])
    tx_imp[:, 0] = np.where(tx_imp[:, 0] == -999, np.argmax(counts), tx_imp[:, 0]) 

    # Imputation of data for jet_num = 0 and jet_num = 1
    tx_imp[jet_num_idx[0], :] = np.where(tx_imp[jet_num_idx[0], :] == -999, 0, tx_imp[jet_num_idx[0], :])
    tx_imp[jet_num_idx[1], :] = np.where(tx_imp[jet_num_idx[1], :] == -999, 0, tx_imp[jet_num_idx[1], :])

    return tx_imp



def variance(data, thresh):
    v_vector = np.var(data, axis=0)
    index_keep = np.where(v_vector > thresh)
    new_data = data[:, index_keep[0]]
    
    return new_data



def correlation(data, thresh_corr):
    corr_mat = np.empty([data.shape[1], data.shape[1]])
    for i in range(data.shape[1]):
        for j in range(i):
            if i != j:
                corr_mat[i, j] = np.corrcoef(data[:, i], data[:, j])[0, 1]

    index_out1 = np.unique(np.where(corr_mat > 0.8)[0])
    index_out2 = np.unique(np.where(corr_mat > 0.8)[1])
    all_idx = range(data.shape[1])

    if len(index_out1) > len(index_out2):
        new_data = data[:, np.setdiff1d(all_idx, index_out1)]
    else:
        new_data = data[:, np.setdiff1d(all_idx, index_out2)]
    
    return new_data



def remove_outliers(tx, y, std_limit = 4):

    num_datapoints = np.shape(tx)[0]
    num_feat = np.shape(tx)[1]
    indices = np.indices((1,num_datapoints))

    standardized = standardize(tx)
    number_outliers = np.zeros((1,num_feat))
    index_outliers = []

    for ii in range(num_feat):    
        pos_outlier = standardized[:,ii]>std_limit
        neg_outlier = standardized[:,ii]<-std_limit
        number_outliers[0,ii] = np.sum(pos_outlier) + np.sum(neg_outlier)
    
        for jj in range(num_datapoints):
            if (pos_outlier[jj] == True or neg_outlier[jj] == True) and jj not in index_outliers:
                index_outliers.append(jj)

    standardized_outliers_removed = standardized[np.setdiff1d(indices,index_outliers)]
    y_std = y[np.setdiff1d(indices,index_outliers)]
    
    return standardized_outliers_removed, y_std