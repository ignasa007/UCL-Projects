import numpy as np

from q3_f import hessian
from q3_utils import cauchy_point, unconstrained_min_along_g, full_step

def dogleg_direction(x_k, delta):

    # compute the Cauchy point, p_c
    p_c = cauchy_point(x_k, delta)
    # if the Hessian is not PD, return the Cauchy point
    if not np.all(np.linalg.eigvals(hessian(x_k)) > 0):
        return p_c

    # compute the unconstrained minimizer along the negative gradient
    p_u = unconstrained_min_along_g(x_k)
    # if x_k+p_u lies outside the trust region, return the Cauchy point
    if np.linalg.norm(p_u) >= delta:
        return p_c
    
    # compute the full-step solution
    p_b = full_step(x_k)
    # if x_k+p_b lies inside the trust region, pick that
    if np.linalg.norm(p_b) <= delta:
        return p_b

    # else solve for \tau, which gives us the optimal direction between p_u to p_b
    a = np.square(np.linalg.norm(p_b-p_u))
    b = 2 * (p_b-p_u).T@p_u
    c = np.square(np.linalg.norm(p_u)) - delta**2
    discriminant = b**2 - 4*a*c

    # solutions of the quadratic problem; note that these are the roots of \tau - 1
    r1, r2 = (-b + discriminant**0.5) / (2*a), (-b - discriminant**0.5) / (2*a)
    if 0 <= r1 <= 1:
        return p_u + r1 * (p_b-p_u)
    if 0 <= r2 <= 1:
        return p_u + r2 * (p_b-p_u)
    
    raise AssertionError(
        'Failed to find a optimum on the dogleg:\n'
        f'\tx_k = {x_k}, delta = {delta}\n'
        f'\tp_u = {p_u}, p_b = {p_b}\n'
        f'\ta = {a}, b = {b}, c = {c}, discriminant = {discriminant}\n'
        f'\tr1 = {r1}, r2 = {r2}'
    )