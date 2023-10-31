#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMP0078 Coureswork 1
Part 1.1 - Linear regression

Isaac Watson
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(sigma, N):
    """
    Function to generate X data from a uniform distribution
    and corresponding y data with normal errors.

    Parameters
    ----------
    sigma : standard deviation of normal errors
    N : number of data points

    Returns
    -------
    X : X data
    y : corresponding y data

    """
    X = np.random.uniform(0,1,N) 
    y = np.sin(2*np.pi*X)**2 + np.random.normal(0,sigma,N)
    return X, y
 
def poly_feature_space(X, k):
    """
    Function to create the polynomial feature space.
    The function loops through all x applying polynomial basis of order k
    to each data point.
    The matrix rows correpsond to the ith data point.
    The matrix columns correspond to the jth polynomial basis function.

    Parameters
    ----------
    X : X data points
    order : polynomial basis order

    Returns
    -------
    phi : polynomial feature space of order k

    """
    # initialise empty feature space
    phi = np.zeros([len(X), k])
    
    # perform iterative assignment of matrix values
    for i in range(len(X)):
        for j in range(k):
            # allocate x_i^j to position i,j
            phi[i,j]+=X[i]**j
    return phi

def optimal_weights(phi, y):
    """
    Function to obtain the optimal weights.
    The function performs the matrix operation (phi^T*phi)^-1*phi^T*y and
    returns the result.

    Parameters
    ----------
    phi : feature space
    y : y vector

    Returns
    -------
    Vector of optimal weights

    """
    # utilise numpys inverse function to compute optimal weights
    return np.linalg.inv(phi.T@phi)@phi.T@y

def fit_poly(w, X):
    """
    Function to produce fitted y data applying the optimal weights to the corresponding
    basis function.

    Parameters
    ----------
    w : optimal weights
    X : X data

    Returns
    -------
    Vector of fitted y values

    """
    # list comprehension to return an array of y values according to the optimal 
    # linear combination of polynomial basis functions
    return np.array([np.sum(np.array([w[i]*(x**i) for i in range(len(w))])) for x in X])

def sin_feature_space(X, k):
    """
    Function to create the sine feature space.
    The function loops through all x applying sine basis of order k
    to each data point.
    The matrix rows correspond to the ith data point.
    The matrix columns correspond to the jth sine basis function.

    Parameters
    ----------
    X : X data points
    order : sine basis order

    Returns
    -------
    phi : sine feature space of order k

    """
    # initialise empty feature space
    phi = np.zeros([len(X), k])
    
    # perform iterative assignment of matrix values
    for i in range(len(X)):
        for j in range(k):
            # allocate sin((j=1)*pi*x_i) to position i,j
            phi[i,j]+=np.sin((j+1)*np.pi*X[i])
    return phi


def fit_sin(w, X):
    """
    Function to produce fitted y data applying the optimal weights to the corresponding
    basis function.

    Parameters
    ----------
    w : optimal weights
    X : X data

    Returns
    -------
    Vector of fitted y values

    """
    # list comprehension to return an array of y values according to the optimal 
    # linear combination of sine basis functions
    return np.array([np.sum(np.array([w[i]*(np.sin((i+1)*np.pi*x)) for i in range(len(w))])) for x in X])

# =============================================================================
# Question 1
# =============================================================================
# =============================================================================
# a
# =============================================================================

# sample data for q1
SX1 = np.array([1, 2, 3, 4])
Sy1 = np.array([3, 2, 0, 5])

# data for plotting full curve
x_axis = np.linspace(0,5,1000)

# intitialise empty feature space store and weights store
f_space_set1 = []
weights_set1 = []

# empty mse dictionary
mse = {'Basis 1' :None, 'Basis 2': None, 'Basis 3': None, 'Basis 4': None}

# loop to generate a feature space and optimal set of weights for each basis vector
# {1}, {1, x}, {1, x, x^2}, {1, x, x^2, x^3}
for i in range(4):
    
    # create feature space and append to store
    phi = poly_feature_space(SX1, i+1)
    f_space_set1.append(phi)
    
    # create weight vector and append to store
    w = optimal_weights(phi, Sy1)
    weights_set1.append(w)
    
    # calculate fitted y values from the sample X data to use for mse calculation
    Sy1_fitted = fit_poly(w, SX1)
    mse["Basis "+str(i+1)] = np.average((Sy1-Sy1_fitted)**2)
    
    # calculate y fitted for 'all' x for plotting
    y_fitted = fit_poly(w, x_axis)
    plt.plot(x_axis, y_fitted, label="Basis order "+str(i+1))
    
# show plot overlaying sample data   
plt.scatter(SX1,Sy1)
plt.legend()
plt.show()

# =============================================================================
# b
# =============================================================================

# Equations of fitted curves
"""
{1} => y=2.5
{1, x} => y=1.5+0.4x
{1, x, x^2} => y=9-1.7x+1.5x^2
{1, x, x^2, x^3} => y=-5+15.17x-8.5x^2+1.33x^3
"""

# =============================================================================
# c
# =============================================================================

# show mse dictionary
print(mse)
"""
Basis 1: 3.25
Basis 2: 3.0500000000000003
Basis 3: 0.7999999999999992
Basis 4: 2.0896272251778886e-23
"""

# =============================================================================
# Question 2
# =============================================================================
# =============================================================================
# a. i
# =============================================================================

sigma = 0.07
SX, Sy = generate_data(sigma, 30)

x_axis = np.linspace(0,1,1000)
y_true = np.sin(2*np.pi*x_axis)**2 

plt.figure()
plt.scatter(SX, Sy)
plt.plot(x_axis, y_true)

# =============================================================================
# a. ii
# =============================================================================

# produce feature spaces looping through X data and applying ith order polynomials
f_space_set2 = []
weight_set2 = []
Smse = []

orders = [2, 5, 10, 14, 18]
for s in range(len(orders)):
    
    phi = poly_feature_space(SX, orders[s])
    f_space_set2.append(phi)
    
    w = optimal_weights(phi, Sy)
    weight_set2.append(w)
    
    # mse calculation
    y_fitted = fit_poly(w, x_axis)
    
    # plot data
    plt.figure()
    plt.title(f"Polynomial basis set k={orders[s]} fit")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.plot(x_axis, y_fitted)
    plt.scatter(SX, Sy)
    plt.ylim(-0.5,1.5)
    plt.show()
 
# =============================================================================
# b    
# =============================================================================

f_space_set3 = []
weights_set3 = []

Smse = []

orders18 = np.arange(0,18)

for i in orders18:

    phi = poly_feature_space(SX, i+1)
    f_space_set3.append(phi)
    
    w = optimal_weights(phi, Sy)
    weights_set3.append(w)
    
    Sy_fitted = fit_poly(w, SX)
    Smse.append(np.average((Sy-Sy_fitted)**2))
    
log_Smse = np.log(Smse)

plt.figure()
plt.title("MSE for "r"$x^k$"" basis set - sample")
plt.xlabel("Dimension (k)")
plt.ylabel("ln(MSE)")
plt.plot(orders18, log_Smse)
plt.show()

# =============================================================================
# c
# =============================================================================

TX, Ty = generate_data(sigma, 1000)

Tmse = []

for s in orders18: 
    Ty_fitted = fit_poly(weights_set3[s], TX)
    Tmse.append(np.average((Ty-Ty_fitted)**2))
    
log_Tmse = np.log(Tmse)

plt.figure()
plt.title("MSE for "r"$x^k$"" basis set - test")
plt.xlabel("Dimension (k)")
plt.ylabel("ln(MSE)")
plt.plot(orders18, log_Tmse)
plt.show()

# =============================================================================
# d
# =============================================================================
        
total_Smse = np.array([0]*18, float)
total_Tmse = np.array([0]*18, float)

runs = 100
for r in range(runs):
    SX, Sy = generate_data(sigma, 30)
    TX, Ty = generate_data(sigma, 1000)
    
    Smse = []
    Tmse = []
    
    for s in range(0,18): 
        phi = poly_feature_space(SX, s+1)
        
        w = optimal_weights(phi, Sy)
        
        Sy_fitted = fit_poly(w, SX)
        Smse.append(np.average((Sy-Sy_fitted)**2))

        Ty_fitted = fit_poly(w, TX)
        Tmse.append(np.average((Ty-Ty_fitted)**2))
        
    total_Smse += np.array(Smse)
    total_Tmse += np.array(Tmse)
    
logavg_Smse = np.log(total_Smse/runs)
logavg_Tmse = np.log(total_Tmse/runs)

plt.figure()
plt.title("100 run average MSE for "r"$x^k$"" basis set")
plt.xlabel("Dimension (k)")
plt.ylabel("ln(MSE)")
plt.plot(np.arange(0,18), logavg_Smse, label='Sample') 
plt.plot(np.arange(0,18), logavg_Tmse, label='Test') 
plt.legend()
plt.show()


# =============================================================================
# 3
# =============================================================================

sin_f_space_set = []
sin_weights_set = []

sin_Smse = []

orders18 = np.arange(0,18)

for i in orders18:
    phi = sin_feature_space(SX, i+1)
    sin_f_space_set.append(phi)
    
    w = optimal_weights(phi, Sy)
    sin_weights_set.append(w)
    
    Sy_fitted = fit_sin(w, SX)
    sin_Smse.append(np.average((Sy-Sy_fitted)**2))
    
logsin_Smse = np.log(sin_Smse)

plt.figure()
plt.title("MSE for sin(k*$\pi*x) basis set - sample")
plt.xlabel("Dimension (k)")
plt.ylabel("ln(MSE)")
plt.plot(orders18, logsin_Smse)
plt.show()

TX, Ty = generate_data(sigma, 1000)

sin_Tmse = []

for s in orders18: 
    Ty_fitted = fit_sin(sin_weights_set[s], TX)
    sin_Tmse.append(np.average((Ty-Ty_fitted)**2))
    
logsin_Tmse = np.log(sin_Tmse)

plt.figure()
plt.title("MSE for sin(k"r"$\pi$""x) basis set - test")
plt.xlabel("Dimension (k)")
plt.ylabel("ln(MSE)")
plt.plot(orders18, log_Tmse)
plt.show()
 
total_Smse = np.array([0]*18, float)
total_Tmse = np.array([0]*18, float)

runs = 100
for r in range(runs):
    SX, Sy = generate_data(sigma, 30)
    TX, Ty = generate_data(sigma, 1000)
    
    Smse = []
    Tmse = []
    
    for s in range(0,18): 
        phi = sin_feature_space(SX, s+1)
        
        w = optimal_weights(phi, Sy)
        
        Sy_fitted = fit_sin(w, SX)
        Smse.append(np.average((Sy-Sy_fitted)**2))

        Ty_fitted = fit_sin(w, TX)
        Tmse.append(np.average((Ty-Ty_fitted)**2))
        
    total_Smse += np.array(Smse)
    total_Tmse += np.array(Tmse)
    
logavgsin_Smse = np.log(total_Smse/runs)
logavgsin_Tmse = np.log(total_Tmse/runs)

plt.figure()
plt.title("100 run average MSE for sin(k*$\pi*x) basis set")
plt.xlabel("Dimension (k)")
plt.ylabel("ln(MSE)")
plt.plot(np.arange(0,18), logavgsin_Smse, label='Sample') 
plt.plot(np.arange(0,18), logavgsin_Tmse, label='Test') 
plt.legend()
plt.show()
    
    
    
    
    
    
    
    
    