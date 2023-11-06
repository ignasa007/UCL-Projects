#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMP0078 Coureswork 1
Part 1.2 - Filtered Boston housing and kernels

Isaac Watson
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
    
def optimal_weights(phi, y):
    """
    Function to obtain the optimal weights.
    The function performs the matrix operation (phi^T*phi)^-1*phi^T*y and
    returns the result. 
    Works for simple linear regresison also replacing phi with X.

    Parameters
    ----------
    phi : feature space or X data
    y : y vector

    Returns
    -------
    Vector of optimal weights

    """
    # utilise numpys inverse function to compute optimal weights
    return np.linalg.inv(phi.T@phi)@phi.T@y


data = pd.read_csv("boston_housing.csv")
X = data.drop(columns='MEDV').to_numpy()
y = data['MEDV'].to_numpy()
y = y.reshape([len(y),1])

results = pd.DataFrame()

# =============================================================================
# a
# =============================================================================

mse_total_train_const = 0
mse_total_test_const = 0

for i in range(20):
 
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.66) 
    # define constant X data
    ones_train = np.ones([len(X_train), 1])
    ones_test = np.ones([len(X_test), 1])
    
    w = optimal_weights(ones_train, y_train)
    pred_train_const = ones_train@w
    pred_test_const = ones_test@w
    
    mse_total_train_const += np.average((y_train-pred_train_const)**2)
    mse_total_test_const += np.average((y_test-pred_test_const)**2)
    
mse_avg_train_const = mse_total_train_const/20
mse_avg_test_const = mse_total_test_const/20
        
# =============================================================================
# b  
# =============================================================================

# The constant obtained in question (a) is the average value of the y data

# =============================================================================
# c
# =============================================================================

ones_X = np.ones([len(X), 1])
X_bias = np.append(ones_X, X, axis=1)
    
mse_col_dict = {col: None for col in data.columns[:-1]}

for col in range(1,13):
    
    mse_total_train_col = 0
    mse_total_test_col = 0
    
    for i in range(20):
        
        X_col_train, X_col_test, y_train, y_test = train_test_split(X_bias[:,[0,col]], 
                                                                    y, test_size=0.66) 
        
        w = optimal_weights(X_col_train, y_train)
        pred_train_col = X_col_train@w
        pred_test_col = X_col_test@w
            
        mse_total_train_col += np.average((y_train-pred_train_col)**2)
        mse_total_test_col += np.average((y_test-pred_test_col)**2)
        
    mse_col_dict[data.columns[col-1]] = {'Train': mse_total_train_col/20, 
                                         'Test': mse_total_test_col/20}
    
# =============================================================================
# d
# =============================================================================

mse_total_train = 0
mse_total_test = 0

for i in range(20):
 
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.66) 
    ones_train = np.ones([len(X_train), 1])
    X_train = np.append(ones_train, X_train, axis=1)
    ones_test = np.ones([len(X_test), 1])
    X_test = np.append(ones_test, X_test, axis=1)
    
    w = optimal_weights(X_train, y_train)
    pred_train = X_train@w
    pred_test = X_test@w
    
    mse_total_train += np.average((y_train-pred_train)**2)
    mse_total_test += np.average((y_test-pred_test)**2)
    
mse_avg_train = mse_total_train/20
mse_avg_test = mse_total_test/20
