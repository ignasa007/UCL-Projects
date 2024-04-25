import numpy as np

@np.vectorize
def f(x, y):
    f_xy = 100*np.square(np.square(y)+np.square(x)-1) + np.square(1-x)
    return f_xy if f_xy.size > 1 else f_xy.squeeze().item()

def gradient(x):
    x, y = x.flatten()
    df_xy = np.array([
        [100*(4*x**3+4*x*y**2-4*x) + 2*(x-1)], 
        [100*(4*y**3+4*y*x**2-4*y)]
    ])
    return df_xy

def hessian(x):
    x, y = x.flatten()
    d2f_xy = np.array([
        [100*(12*x**2+4*y**2-4) + 2, 100*(8*x*y)],
        [100*(8*x*y), 100*(12*y**2+4*x**2-4)]
    ])
    return d2f_xy

x_opt = np.array([[1.], [0.]])

# print(hessian(np.array([[0.], [0.]])))