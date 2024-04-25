import numpy as np
from Q3d import EMForMultivariateBernoulli, plot_params

# Load the dataset
dataset = np.loadtxt('binarydigits.txt')

# Loop over the values of K, running EM for each of them
Ks = (2, 3, 4, 7, 10)

for run in range(5):
    for K in Ks:
        em = EMForMultivariateBernoulli()
        P, pi, R, log_likelihoods = em(K=K, X=dataset, n_iterations=15, eps=1e-1)
        for k in range(K):
            plot_params(params=P[k].reshape(8, 8), k=k, pi_k=pi[k], 
                fn=f'assets/em-repeated/run={run+1}/K={K}/k={k+1}.png')