import numpy as np

from q3_f import f, gradient
from dogleg import dogleg_direction
from q3_utils import reduction

def trust_region(x_0, delta_max=3, eta=0.2, tol=1e-6, max_iter=100):

    assert 0 <= eta < 0.25
    print('Trust region search with dogleg approximation')

    k = 0
    x_k = x_0; delta_k = delta_max / 2
    xs = [x_k]
    stop_cond = False

    while not stop_cond and k < max_iter:
        
        k += 1
        p_k = dogleg_direction(x_k, delta_k)
        rho_k = reduction(x_k, p_k)
        
        if rho_k < 1/4:
            delta_k = delta_k/4
        elif rho_k > 3/4 and np.linalg.norm(p_k) == delta_k:
            delta_k = min(2*delta_k, delta_max)
        
        if rho_k > eta:
            x_k = x_k + p_k
            xs.append(x_k)    
            stop_cond = (np.linalg.norm(gradient(x_k).flatten(), float('inf')) < tol*(1+abs(f(*x_k.flatten()))))
        elif delta_k < 1e-6 * delta_max:
            stop_cond = True

    x_min = x_k
    f_min = f(*x_k.flatten())

    print(f'After {len(xs)-1} iterations, x_min = {np.round(x_min, 6).flatten().tolist()}, f(x_min) = {f_min}')

    errors = []
    for x in xs[:-1]:
        errors.append(np.linalg.norm(xs[-1]-x))
    errors = np.array(errors)

    return xs, errors