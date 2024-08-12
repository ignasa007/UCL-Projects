import numpy as np

from q3_f import f, gradient, hessian

def model(x_k, p):

    return f(*x_k.flatten()) + gradient(x_k).T@p + 0.5 * p.T@hessian(x_k)@p

def reduction(x_k, p):

    return (f(*x_k.flatten()) - f(*(x_k+p).flatten())) / (model(x_k, np.zeros_like(x_k)) - model(x_k, p))

def cauchy_point(x_k, delta):

    g_k = gradient(x_k)
    B_k = hessian(x_k)
    p_s = -delta * g_k/np.linalg.norm(g_k)
    tau_k = 1 if g_k.T@B_k@g_k <= 0 else min(1, np.linalg.norm(g_k)**3 / (delta*g_k.T@B_k@g_k))
    
    return tau_k * p_s 

def unconstrained_min_along_g(x_k):

    g_k = gradient(x_k)
    B_k = hessian(x_k)

    return -(g_k.T@g_k)/(g_k.T@B_k@g_k) * g_k

def full_step(x_k):

    g_k = gradient(x_k)
    B_k = hessian(x_k)

    return -np.linalg.pinv(B_k)@g_k