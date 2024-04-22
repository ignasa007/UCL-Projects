import numpy as np

class LinearKernel:

    def __init__(self):

        self.radius_sq = 1

    def __call__(self, *args):

        return self.compute(*args)

    def compute(self, x_rows, x_cols):

        cosine_distances = x_rows @ x_cols.T \
            / np.linalg.norm(x_rows, axis=1, keepdims=True) \
            / np.linalg.norm(x_cols, axis=1, keepdims=True).T
        assert cosine_distances.shape == (len(x_rows), len(x_cols))
        
        return cosine_distances
    
class GaussianKernel:

    def __init__(self, gamma):

        self.radius_sq = 1
        self.gamma_sq = gamma**2

    def __call__(self, *args):

        return self.compute(*args)

    def compute(self, x_rows, x_cols):

        distances = np.square(np.linalg.norm(x_rows, axis=1, keepdims=True)) \
            + np.square(np.linalg.norm(x_cols, axis=1, keepdims=True)).T \
            - 2 * x_rows @ x_cols.T
        distances = np.exp(-self.gamma_sq*distances)
        assert distances.shape == (len(x_rows), len(x_cols))
        
        return distances