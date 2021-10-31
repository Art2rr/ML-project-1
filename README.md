# ML-project-1

The project consists of implementation of following machine learning methods: least squares gradient descent, ridge regression and logistic regresion gradient descent. These methods are used to create a machine learing model for the ML higgs challenge (Project 1 of CS-433 EPFL course). The challenge details: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs.
The project includes the code for implementation of ML methods, data processing, hyperparametr optimisation and classification prediction using constructed models. The whole process can be accessed by running _run.py_ file (by default this constructs the model with pre-optimised hyperparameters).

## Data Processing
Data processing consists of grouping the data based on their jet number in order to be able to construct individual models for each group. Grouped data is standarised and the outliers are removed. The features that do not carry information (low variance or high correlation with other features) are removed

## ML models
The availible code allows for construction of following ML models: least square gradient descent, least square stochastic gradient descent, ridge regression, logistic regression gradiend descent and logistic regression stochastic gradient descent. For each method a function returns the optimised weights of the features and the loss value for these values. The loss function for least squares and ridge regression is mean squared error and the loss function for logistic regression is the negative log likelihood (for binary classification between {0,1})

## Hyperparametr Optimisation
all of the ML methods can be optimised with regards to the degree of the polynomial to which the features are raised. Additionally, ridge regression and logistic regresion can be optimised with respect to reguralisation parameter lambda. The two hyperparameters can be optimised using k-fold cross validation on the training set. Implemented functions allow for testing different values of lambda and degree at the same time.

## Classification Prediction
the _run.py_ script can be used to generate the prediction of classification for a test data (by default _test.csv_) given the pre-classified training data (by default _train.csv_). imput and output data are classifed between {-1,1}.

## Files

Individual functions are availible in following files:

* **implementations.py** - functions constructing ML models

* **helper_GD_SGD.py** - functions calculating gradient for gradient descent

* **helpers_LR.py** - functions calculating gradient and loss function for logistic regression. function calculating sigmoid function

* **data_processing.py** - functions standarisation of data, grouping them by jet number and removing outliers. Also includes functions for removing features based on the variance as well as highly correlated features

* **hyperparameters.py** - functions implementing cross-validation for ML methods used for hyperparametrs optimisation (reguralisation parameter lambda and polynomial degree).

* **proj1_helpers** - functions to load data and save predictions. function to predict classification.

* **run.py** - script allowing to process data, build and optimise the ML model, and use it to predict the classification