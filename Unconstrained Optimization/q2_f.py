import numpy as np

@np.vectorize
def f(x, y):
    if x <= 0.:
        return float('inf')
    f_xy = np.square(y+np.log(x)) + np.square(y-x)
    return f_xy if isinstance(f_xy, float) or f_xy.size > 1 else f_xy.squeeze()[0]

def gradient(x):
    x, y = x.flatten()
    if x <= 0.:
        raise ValueError('x cannot be <= 0.') 
    df_xy = np.array([
        [2*((y+np.log(x))/x - (y-x))], 
        [2*((y+np.log(x)) + (y-x))]
    ])
    return df_xy

def hessian(x):
    x, y = x.flatten()
    if x <= 0.:
        raise ValueError('x cannot be <= 0.')
    d2f_xy = np.array([
        [2*(1+(1-np.log(x)-y)/x**2), 2*(-1+1/x)],
        [2*(-1+1/x), 4]
    ])
    return d2f_xy

x_opt = np.array([[0.56714329], [0.56714329]])