from proj1_helpers import *
from implementations import *

# Download training data
DATA_TRAIN_PATH = 'train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Create groups according to jet number
jet_num_idx = grouping(tX)

# Imputation
tx = imputation (tX, jet_num_idx)

# Feature selection through Variance Threshold
#---------------- 0 Jet -------------------
data = tx[jet_num_idx[0][0], :]
thresh = 0
data_j0 = variance(data, thresh)

#---------------- 1 Jet -------------------
data = tx[jet_num_idx[1][0], :]
data_j1 = variance(data, thresh)

#---------------- 2 Jet -------------------
data = tx[jet_num_idx[2][0], :]
data_j2 = variance(data, thresh)

#---------------- 3 Jet -------------------
data = tx[jet_num_idx[3][0], :]
data_j3 = variance(data, thresh)


# Feature selection through Correlation Coefficient
#---------------- 0 Jet -------------------
thresh_corr = 0
data_j0 = variance(data_j0, thresh_corr)

#---------------- 1 Jet -------------------
data_j1 = variance(data_j1, thresh_corr)

#---------------- 2 Jet -------------------
data_j2 = variance(data_j2, thresh_corr)

#---------------- 3 Jet -------------------
data_j3 = variance(data_j3, thresh_corr)


# Standardization
#---------------- 0 Jet -------------------
data_j0 = standardize(data_j0)

#---------------- 1 Jet -------------------
data_j1 = standardize(data_j1)

#---------------- 2 Jet -------------------
data_j0 = standardize(data_j0)

#---------------- 3 Jet -------------------
data_j0 = standardize(data_j0)