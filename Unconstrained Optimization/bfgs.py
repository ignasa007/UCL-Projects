import numpy as np

from q5_f import f, gradient
from line_search import line_search_wolfe

def bfgs(x_0, tol=1e-6):
    
    k = 0; x_k = x_0.copy(); xs = [x_k]
    I_2 = np.eye(x_0.size); H_k = I_2.copy()

    while np.linalg.norm(gradient(x_k)) > tol:
        p_k = -H_k@gradient(x_k)
        alpha_k = line_search_wolfe(f, gradient, x_k, p_k, 1, c2=0.1)
        k += 1
        x_k = x_k + alpha_k*p_k
        s_k = x_k - xs[-1]
        y_k = gradient(x_k) - gradient(xs[-1])
        rho_k = 1 / (s_k.T@y_k)
        H_k = (I_2-rho_k*s_k@y_k.T) @ H_k @ (I_2-rho_k*y_k@s_k.T) + rho_k*s_k@s_k.T
        xs.append(x_k)

    x_min = x_k
    f_min = f(*x_k.flatten())

    print(f'After {len(xs)-1} iterations, x_min = {np.round(x_min, 6).flatten().tolist()}, f(x_min) = {f_min}')

    errors = []
    for x in xs[:-1]:
        errors.append(np.linalg.norm(xs[-1]-x))
    errors = np.array(errors)

    return xs, errors