import numpy as np
import matplotlib.pyplot as plt

from ssm_kalman import run_ssm_kalman

def maximisation(X, Y, V, Vj):

    '''
    Execute the M-step of the EM algorithm, computing the parameters maximizing the 
    expectation of the log-likelihood under the distribution q.
    :param X: dataset of shape (D, T), where T is the number of time steps and D 
        is the dimension of the observations.
    :param Y: array of shape (K, T) with posterior mean estimates, E[y_t|x_{1:T}], 
        where K is the dimesion of the latents.
    :param V: array of shape (T, K, K) with posterior covariance estimates, 
        Cov[y_t,y_t|x_{1:T}].
    :param Vj: array of shape (T, K, K) with posterior cross covariance estimates, 
        Cov[y_t,y_{t-1}|x_{1:T}].
    :returns:
    C_hat: array of shape (K, K), equal to the estimate for C.
    A_hat: array of shape (K, K), equal to the estimate for A.
    R_hat: array of shape (K, K), equal to the estimate for R.
    Q_hat: array of shape (K, K), equal to the estimate for Q.
    '''

    # \sum_{t=1}^T x_t@E(y_t).T
    mat1 = X@Y.T
    # \sum_{i=1}^T E(y_t)@E(y_t).T
    mat2 = np.expand_dims(Y.T, 2) @ np.expand_dims(Y.T, 1)
    # \sum_{t=1}^T E(y_t@y_t.T) = \sum_{t=1}^T Cov(y_t,y_t) + E(y_t)@E(y_t).T
    mat3 = np.sum(V + mat2, axis=0)
    # \sum_{t=1}^{T-1} E(y_t@y_t.T) = \sum_{t=1}^{T-1} Cov(y_t,y_t) + E(y_t)@E(y_t).T
    mat4 = np.sum(V[:-1] + mat2[:-1], axis=0)
    # \sum_{t=2}^T E(y_t@y_t.T) = \sum_{t=2}^T Cov(y_t,y_t) + E(y_t)@E(y_t).T
    mat5 = np.sum(V[1:] + mat2[1:], axis=0)
    # \sum_{t=2}^T E(y_t@y_{t-1}.T) = \sum_{t=2}^T Cov(y_t,y_{t-1}) + E(y_t)@E(y_{t-1}).T
    mat6 = np.sum(Vj[:-1] + np.expand_dims(Y.T[1:], 2) @ np.expand_dims(Y.T[:-1], 1), axis=0)
    
    A_hat = mat6 @ np.linalg.inv(mat4)
    Q_hat = (mat5 - mat6@A_hat.T) / (X.shape[1]-1)
    C_hat = mat1 @ np.linalg.inv(mat3)
    R_hat = (X@X.T - mat1@C_hat.T) / (X.shape[1])

    return A_hat, Q_hat, C_hat, R_hat

if __name__ == '__main__':

    # Load the dataset
    X = np.loadtxt('ssm_spins.txt').T

    # Define the covariance matrix, Q_1, of the initial hidden state, y_1
    Q_init = np.identity(4)
    # Define the total number of runs and the total number of EM iterations in each run
    n_runs, n_iterations = 10, 100
    # Define an (n_runs, n_iterations+1) sized array to collect the results of each EM run 
    all_log_likelihoods = np.zeros((n_runs, n_iterations+1))

    for run in range(n_runs):

        # Sample the initial hidden state, y_1
        y_init = np.random.multivariate_normal(np.zeros(4), Q_init)
        # Initialize the parameters governing transitions in the latent space, A
        A = np.random.uniform(low=-5, high=5, size=(4, 4))
        # Initialize a random PSD matrix, Q, for the random transition
        Q = np.random.uniform(low=-5, high=5, size=(4, 4)); Q = Q.T@Q
        # Initialize the parameters governing emission of the observations, C
        C = np.random.uniform(low=-5, high=5, size=(5, 4))
        # Initialize a random PSD matrix, R, for the random emission
        R = np.random.uniform(low=-5, high=5, size=(5, 5)); R = R.T@R

        for iteration in range(n_iterations+1):
            # Run the E-step of the algorithm
            Y, V, Vj, L = run_ssm_kalman(X, y_init.copy(), Q_init.copy(), A, Q, C, R, mode='smooth')
            # Record the log likelihood as the sum of log-conditional-likelihoods for x_1, ..., x_T
            all_log_likelihoods[run, iteration] = np.sum(L)
            # Run the M-step of the algorithm
            A, Q, C, R = maximisation(X, Y, V, Vj)

    # Remove the log-likelihood corresponding to the initial random parameters 
    all_log_likelihoods = all_log_likelihoods[:, 1:]
    first_50_log_ls = all_log_likelihoods[:, :50]
    iter_50_log_ls = all_log_likelihoods[:, 50]

    # Plot the log-likelihoods
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for i, log_likelihoods in enumerate(first_50_log_ls, 1):
        axs.plot(range(1, 51), log_likelihoods/1000, color=f'C{i}')
    axs.set_xticks(range(10, 51, 10)) 
    axs.set_ylim(first_50_log_ls.min()/1000-0.1, first_50_log_ls.max()/1000+0.1)
    axs.set_xlabel('EM Iteration', size=12)
    axs.set_ylabel(r'Log-likelihood $\times 10^{-3}$', size=12)
    axs.grid()
    fig.tight_layout()
    plt.savefig(f'assets/kalman/em-50.png')
    plt.close(fig)

    # Plot the final log-likelihood in each run
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.bar(range(1, n_runs+1), iter_50_log_ls/1000, color='blue')
    axs.set_ylim(iter_50_log_ls.min()/1000-0.01, iter_50_log_ls.max()/1000+0.01)
    axs.set_xlabel('Run', size=12)
    axs.set_ylabel(r'Log-likelihood $\times 10^{-3}$', size=12)
    axs.grid()
    fig.tight_layout()
    plt.savefig(f'assets/kalman/em-bar.png')
    plt.close(fig)

    # Plot the standard deviation of the log-likelihoods
    print(np.std(iter_50_log_ls))
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(np.std(all_log_likelihoods, axis=0), color='blue')
    axs.set_xticks(np.arange(0, n_iterations+1, 10))
    axs.set_xlabel('EM Iteration', size=12)
    axs.set_ylabel('StdDev(log-likelihood)', size=12)
    axs.grid()
    fig.tight_layout()
    plt.savefig(f'assets/kalman/em-std.png')
    plt.close(fig)

    # Plot the range of the log-likelihoods
    print(np.ptp(iter_50_log_ls, axis=0))
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(np.ptp(all_log_likelihoods, axis=0), color='blue')
    axs.set_xticks(np.arange(0, n_iterations+1, 10))
    axs.set_xlabel('EM Iteration', size=12)
    axs.set_ylabel('Range(log-likelihood)', size=12)
    axs.grid()
    fig.tight_layout()
    plt.savefig(f'assets/kalman/em-range.png')
    plt.close(fig)