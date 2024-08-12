import numpy as np
from utils import expand_dims
from dataset import train_S as S

def kronecker_delta(idx_rows, idx_cols):

    idx_rows, idx_cols = expand_dims(idx_rows, idx_cols)
    kd_matrix = (idx_rows.T == idx_cols).astype(int)

    assert kd_matrix.shape == (idx_rows.size, idx_cols.size)
    return kd_matrix

class MEBKernel:

    def __init__(self, svm_kernel, C):
        
        self.svm_kernel = svm_kernel
        self.C = C
        self.radius_sq = svm_kernel.radius_sq + 1 + 1/C

    def __call__(self, *args):
        
        return self.compute(*args)

    def compute(self, idx_rows, idx_cols):
        
        y_rows, y_cols = expand_dims(S[idx_rows, -1], S[idx_cols, -1])
        assert y_rows.shape == (1, len(idx_rows)) and y_cols.shape == (1, len(idx_cols))
        
        kernel_matrix = (y_rows.T*y_cols) * (self.compute_svm_kernel(idx_rows, idx_cols) + 1) \
            + kronecker_delta(idx_rows, idx_cols) / self.C
        assert kernel_matrix.shape == (len(idx_rows), len(idx_cols))
        
        return kernel_matrix
    
    def compute_svm_kernel(self, idx_rows, idx_cols):

        x_rows, x_cols = expand_dims(S[idx_rows, :-1], S[idx_cols, :-1])
        return self.svm_kernel(x_rows, x_cols)
    
    def init_coreset(self, indices):
        
        self.coreset_precomputations = self.compute(np.arange(len(S)), indices)
        assert self.coreset_precomputations.shape == (len(S), len(indices))
    
    def add_precomputations(self, index):
        
        coreset_size = self.coreset_precomputations.shape[1]
        self.coreset_precomputations = np.hstack((
            self.coreset_precomputations,
            self.compute(np.arange(len(S)), [index])
        ))
        assert self.coreset_precomputations.shape == (len(S), coreset_size+1)

    def remove_precomputations(self, index):

        coreset_size = self.coreset_precomputations.shape[1]
        self.coreset_precomputations = np.delete(self.coreset_precomputations, [index], 1)
        assert self.coreset_precomputations.shape == (len(S), coreset_size-1)