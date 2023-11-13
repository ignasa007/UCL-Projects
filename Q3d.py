import os
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = np.loadtxt('binarydigits.txt')

class EMForMultivariateBernoulli:

    def log_likelihood(self, X, P, pi):
        '''
        Calculate the log-likelihood of the dataset under mixture of multivariate 
        Bernoulli distributions.
        :param X: dataset of shape (N, D), where N is the number of samples and D 
            is the dimension of the samples.
        :param P: parameters of the components arranged as a matrix of shape (D, K),
            where K is the number of components in the mixture.
        :param pi: mixture components' weights.
        :returns:
        :log_likelihood: log-likelihood of the dataset under the mixture.
        '''
        N, D, K = *X.shape, pi.size
        log_likelihood = 0
        for n in range(N):
            x_likelihood = 0
            for k in range(K):
                component_joint = pi[k]
                for d in range(D):
                    component_joint *= (P[k, d]**X[n, d] * (1-P[k, d])**(1-X[n, d]))
                x_likelihood += component_joint
            log_likelihood += np.log(x_likelihood+1e-8)
        return log_likelihood

    def expectation(self, X, P, pi):
        '''
        Execute the E-step of the EM algorithm, computing the expectation of the latents, s^{(n)} 
        corresponding to the observations x^{(n)}.
        :param X: dataset of shape (N, D), where N is the number of samples and D is the 
            dimension of the samples.
        :param P: parameters of the components arranged as a matrix of shape (D, K), where K is 
            the number of components in the mixture.
        :param pi: mixture components' weights.
        :returns:
        R: matrix of shape (N, K) with responsibilities of x^{(n)} towards the mixture components.  
        '''
        N, D, K = *X.shape, pi.size
        R = np.zeros(shape=(N, K))
        for n in range(N):
            for k in range(K):
                R[n, k] = pi[k]
                for d in range(D):
                    R[n, k] *= (P[k, d]**X[n, d] * (1-P[k, d])**(1-X[n, d]))
            R[n] /= np.sum(R[n])
        return R

    def maximisation(self, X, R):
        '''
        Execute the M-step of the EM algorithm, computing the parameters maximizing the 
        expectation of the log-likelihood under the distribution q.
        :param X: dataset of shape (N, D), where N is the number of samples and D 
            is the dimension of the samples.
        :param R: matrix of shape (N, K) with responsibilities of x^{(n)} towards the 
            mixture components.
        :returns:
        P: parameters of the components arranged as a matrix of shape (D, K), where K is 
            the number of components in the mixture.
        pi: mixture components' weights.
        '''
        P = np.divide(R.T @ X, np.clip(R.sum(axis=0), 1e-8, None).reshape(-1, 1))
        pi = np.mean(R, axis=0)
        return P, pi

    def __call__(self, K, X, n_iterations, eps):
        '''
        Run the EM algorithm for a certain number of iterations, terminating early if the
        log-likelihood isn't changing much.
        :param K: number of mixture components.
        :param X: dataset of shape (N, D), where N is the number of samples and D 
            is the dimension of the samples.
        :param n_iterations: maximum number of iterations to run the algorithm for.
        :param eps: threshold below which if the improvement in log-l falls, we stop.
        :returns:
        P: parameters of the components arranged as a matrix of shape (D, K), where K is 
            the number of components in the mixture.
        pi: mixture components' weights.
        R: matrix of shape (N, K) with responsibilities of x^{(n)} towards the mixture components.
        log_likelihoods: log-likelihood for each iteration of the algorithm
        '''
        P = np.random.uniform(size=(K, X.shape[1])); P = np.clip(P, 1e-8, 1-1e-8)
        pi = np.ones(K) / K
        log_likelihoods = [self.log_likelihood(X=X, P=P, pi=pi)]
        for _ in range(n_iterations):
            R = self.expectation(X=X, P=P, pi=pi)
            P, pi = self.maximisation(X=X, R=R); P = np.clip(P, 1e-8, 1-1e-8)
            log_l = self.log_likelihood(X=X, P=P, pi=pi)
            log_likelihoods.append(log_l)
            if log_likelihoods[-1] - log_likelihoods[-2] <= eps:
                break
        return P, pi, R, log_likelihoods
    
def plot_params(params, k, pi_k, fn):
    '''
    Plot the learnt parameters of a component in the mixture.
    :param params: multivariate Bernoulli parameters of the component k.
    :param k: index of the component in the mixture to which the parameters correspond.
    :param pi_k: component weight, i.e., probability of sampling from component k.
    :param fn: filename to save the plot to.
    '''
    fig, axs = plt.subplots(figsize=(4, 5))
    axs.imshow(params, cmap='gray')
    for d in range(params.size):
        i, j = d//8, d%8
        axs.text(j, i, f'{params[i, j]:.2f}', ha='center', va='center', color='orangered')
    axs.set_title(fr'$\pi_{({k+1})} = {pi_k:.6f}$')
    fig.tight_layout()
    plt.axis('off')
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn)
    plt.close(fig)


if __name__ == '__main__':

    # Loop over the values of K, running EM for each of them
    Ks = (2, 3, 4, 7, 10)
    # Store the log-likelihoods for each run to plot later
    all_log_likelihoods = list()
    for K in Ks:
        em = EMForMultivariateBernoulli()
        P, pi, R, log_likelihoods = em(K=K, X=dataset, n_iterations=15, eps=1e-1)
        all_log_likelihoods.append(log_likelihoods)
        for k in range(K):
            plot_params(params=P[k].reshape(8, 8), k=k, pi_k=pi[k], 
                fn=f'assets/em/K={K}/k={k+1}.png')

    # Plot the log-likelihoods from each run to compare across different values of k
    fig, axs = plt.subplots(1, 1, figsize=(4.5, 4))
    for i, (K, log_ls) in enumerate(zip(Ks, all_log_likelihoods)):
        axs.plot(log_ls, marker='o', color=f'C{i}', label=f'K = {K}')
    axs.set_xlabel('Iteration Number')
    axs.set_ylabel('Log-likelihood')
    max_iterations = max((len(log_ls) for log_ls in all_log_likelihoods))
    xticks = range(0, 1+max_iterations//2*2, 2)
    axs.set_xticks(ticks=xticks, labels=xticks)
    axs.grid()
    axs.legend()
    fig.tight_layout()
    plt.savefig(f'assets/em/log_ls.png')
    plt.close(fig)