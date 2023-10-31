#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMP0078 Coureswork 1
Part 1.3 - Kernelised ridge regression

Isaac Watson
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def calculate_alpha(K, y, gamma):
    l = len(K)
    I = np.identity(l)   
    return np.linalg.inv(K+(gamma*l*I))@y

def gaussian_kernel(i, j, sigma):
    dist = np.linalg.norm(i-j)**2
    return np.exp(-dist/(2*sigma**2))

def fit_y(alpha, X, t, sigma):
    return np.sum([alpha[i]*gaussian_kernel(X[i], t, sigma) for i in range(len(alpha))])



# =============================================================================
# a
# =============================================================================

data = pd.read_csv("boston_housing.csv")
X = data.drop(columns='MEDV').to_numpy()
y = data['MEDV'].to_numpy()
y = y.reshape([len(y),1])
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.66) 

gamma_list = [2**i for i in range(-40, -25)]

sigma_power = np.arange(7, 13.5, 0.5)
sigma_list = [2**i for i in sigma_power]

best_gamma = 0
best_sigma = 0

min_error = np.infty

for gamma in gamma_list:
    for sigma in sigma_list:
        
        n = len(X_train)
        fold_size = n//5
        mse = []
        
        for fold in range(5):

            start = fold*fold_size
            end = (fold+1)*fold_size
    
            X_train_cv = np.concatenate((X_train[:start], X_train[end:]), axis=0)
            y_train_cv = np.concatenate((y_train[:start], y_train[end:]), axis=0)
            X_test_cv = X_test[start:end]
            y_test_cv = y_test[start:end]
            
            l = len(X_train_cv)
            kernel = np.zeros([l, l])
            for i in range(l):
                for j in range(l):
                    kernel[i][j] = gaussian_kernel(X_train_cv[i], X_train_cv[j], sigma)
                                    
            alpha = calculate_alpha(kernel, y_train_cv, gamma)
            
            y_pred = []
            for i in range(len(X_test_cv)):
                elem = fit_y(alpha, X_train_cv, X_test_cv[i], sigma) 
                y_pred.append(np.sum(elem))
            
            mse.append(np.average((y_test_cv-y_pred)**2))
            
        mean_mse = np.mean(mse)
        if mean_mse < min_error:
            best_gamma = gamma
            best_sigma = sigma
            min_error = mean_mse
        
        


