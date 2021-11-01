# Methods in machine learning applied to high energy physics

**The final submission for the project can be found in the folder Submission**

The project consists of implementation of following machine learning methods: Linear regression using gradient descent, Linear regression using stochastic gradient descent, Least squares regression using normal equations, Ridge regression using normal equations, Logistic regression using gradient descent, Regularized logistic regression using gradient descent. These methods are used to create a machine learing model for the ML higgs challenge (Project 1 of CS-433 EPFL course). The challenge details: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs.
The project includes the code for implementation of ML methods, data processing, hyperparameter optimisation and classification prediction using constructed models. The model that produced the best predictions can be accessed by running _run.py_ file (by default this constructs the model with pre-optimised hyperparameters). The whole processed can be accessed on the notebook file _project1.ipynb_.

## Data Processing
Data processing consists of grouping the data based on their jet number in order to be able to construct individual models for each group. The data was imputated, features with 0 variance were removed, and finally the data was standardized and outliers with a deviation bigger that 5 std were replaced by that same value.

## ML models
The available code allows for construction of following ML models: Linear regression using gradient descent, Linear regression using stochastic gradient descent, Least squares regression using normal equations, Ridge regression using normal equations, Logistic regression using gradient descent, Regularized logistic regression using gradient descent. For each method a function returns the optimised weights of the features and the loss value for these values. The loss function for the first four implementations is mean squared error and the loss function for logistic regression is the negative log likelihood (for binary classification between {0,1}). 

## Hyperparameter Optimisation
The ML methods can be optimised with regards to the degree of the polynomial to which the features are raised. Additionally, ridge regression and logistic regression can be optimised with respect to reguralisation parameter lambda. The two hyperparameters can be optimised using k-fold cross validation on the training set. Implemented functions allow for testing different values of lambda and degree at the same time.

## Classification Prediction
The _run.py_ script can be used to generate the prediction of classification for a test data (by default _test.csv_) given the pre-classified training data (by default _train.csv_). Input and output data are classifed between {-1,1}.

## Files

Individual functions are availible in following files:

* **implementations.py** - functions constructing ML models

* **helper_GD_SGD.py** - functions calculating gradient for gradient descent

* **helpers_LR.py** - functions calculating gradient and loss function for logistic regression. function calculating sigmoid function

* **data_processing.py** - functions standarisation of data, grouping them by jet number and removing outliers. Also includes functions for removing features based on the variance as well as highly correlated features

* **hyperparameters.py** - functions implementing cross-validation for ML methods used for hyperparametrs optimisation (reguralisation parameter lambda and polynomial degree).

* **proj1_helpers** - functions to load data and save predictions. function to predict classification.

* **run.py** - script allowing to process data, build and optimise the ML model, and use it to predict the classification. The run.py contains the model resulting in the best accuracy, obtained with l2-regularised normal equations.

* **project1.ipynb** - notebook with the optimisation of hyperparameters and plotting the graphs for the paper.

## Visualisation 

In the repository, a folder is included which contains a data matrix with the error and accuracy values for degree = 3 for the different jet groups for a range of lambdas which is defined in the corresponding notebook. The code used to generate the figures in the report and the figures themselves are included here as well. Please note that the matplotlib library was used here, but solely for visualisation purposes.

## Report

For convenience, the pdf file containing the written report is also included in the GitHub repository.
