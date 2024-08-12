import numpy as np

from q2_f import f, gradient, hessian
from line_search import line_search_wolfe

def newton(x_0, tol=1e-3, max_iter=20):

    print('Newton line search with Wolfe conditions')

    n_iter = 1
    x_k = x_0
    xs = [x_k]; alphas = [1]
    stop_cond = False

    while not stop_cond and n_iter <= max_iter:
        n_iter += 1
        p_k = - np.linalg.pinv(hessian(x_k)) @ gradient(x_k)
        p_k = p_k/np.linalg.norm(p_k)
        alpha_k = line_search_wolfe(f, gradient, x_k, p_k, 1)
        x_k = x_k + alpha_k*p_k
        xs.append(x_k); alphas.append(alpha_k)
        stop_cond = (np.linalg.norm(gradient(x_k).flatten(), float('inf')) < tol*(1+abs(f(*x_k.flatten()))))   

    x_min = x_k
    f_min = f(*x_k.flatten())
    print(f'After {len(xs)-1} iterations, x_min = {np.round(x_min, 6).flatten().tolist()}, f(x_min) = {f_min}')

    errors = []
    for x in xs[:-1]:
        errors.append(np.linalg.norm(xs[-1]-x))
    errors = np.array(errors)

    return xs, errors