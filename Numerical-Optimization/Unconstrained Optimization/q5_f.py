import numpy as np

@np.vectorize
def f(x, y):
    f_xy = (x-4*y)**2 + x**4
    return f_xy if not isinstance(f_xy, np.ndarray) or f_xy.size > 1 else f_xy.squeeze().item()

def gradient(x):
    x, y = x.flatten()
    df_xy = np.array([
        [2*(x-4*y) + 4*x**3],
        [-8*(x-4*y)]
    ])
    return df_xy