import numpy as np

from q4_f import f, gradient
from line_search import line_search_wolfe

def ncg(x_0, method, tol=1e-4, max_iter=20, restart=None):

    if method == 'fr':
        print('Nonlinear conjugate gradient algorithm with Fletcher-Reeves updates for the descent direction')
    elif method == 'pr':
        print('Nonlinear conjugate gradient algorithm with Polyak-Ribiere updates for the descent direction')
    else:
        raise ValueError('NCG methods not recognised')
    
    k = 0; x_k = x_0; xs = [x_k]; p_k = -gradient(x_k)
    stop_cond = False
    dirs, steps = [p_k], []

    while not stop_cond and k < max_iter:
        alpha_k = line_search_wolfe(f, gradient, x_k, p_k, 1)
        k += 1
        x_k = x_k + alpha_k*p_k
        df_0, df_1 = gradient(xs[-1]), gradient(x_k)
        if isinstance(restart, int) and k%restart == 0:
            beta_k = 0.
        elif method == 'fr':
            beta_k = np.linalg.norm(df_1)**2 / np.linalg.norm(df_0)**2
        else:
            beta_k = df_1.T@(df_1-df_0) / np.linalg.norm(df_0)**2
        dirs.append(p_k); steps.append(alpha_k)
        p_k = -df_1 + beta_k*p_k
        xs.append(x_k)
        stop_cond = (np.linalg.norm(gradient(x_k), float('inf')) < tol*(1+abs(f(*x_k)))) 

    x_min = x_k
    f_min = f(*x_k.flatten())

    print(f'After {len(xs)-1} iterations, x_min = {np.round(x_min, 6).flatten().tolist()}, f(x_min) = {f_min}')

    errors = []
    for x in xs[:-1]:
        errors.append(np.linalg.norm(xs[-1]-x))
    errors = np.array(errors)

    return xs, errors, dirs, steps