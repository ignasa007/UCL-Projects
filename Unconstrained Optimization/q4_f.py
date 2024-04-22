import numpy as np

@np.vectorize
def f(x, y, a=2, b=0, sigma=0.5, c=4):
    if x <= 0.:
        return float('inf')
    f_xy = x*np.log(x) + np.square(y) + c*np.exp(-(np.square(x-a)+np.square(y-b))/sigma**2)
    return f_xy if f_xy.size > 1 else f_xy.squeeze().item()

def gradient(x, a=2, b=0, sigma=0.5, c=4):
    x, y = x.flatten()
    if x <= 0.:
        raise ValueError('x cannot be <= 0.')
    df_xy = np.array([
        [1 + np.log(x) - 2*c/sigma**2 * (x-a) * np.exp(-(np.square(x-a)+np.square(y-b))/sigma**2)],
        [2*y - 2*c/sigma**2 * (y-b) * np.exp(-(np.square(x-a)+np.square(y-b))/sigma**2)]
    ])
    return df_xy