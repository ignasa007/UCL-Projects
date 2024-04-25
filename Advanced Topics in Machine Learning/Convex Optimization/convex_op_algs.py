import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def generate_problem(n, d, s, std=0.06):

    xsp = 0.5 * (np.random.rand(s // 2) + 1)
    xsn = - 0.5 * (np.random.rand(s // 2) + 1)
    xsparse = np.hstack([xsp, xsn, np.zeros(d - s)])
    random.shuffle(xsparse)
    A = np.random.randn(n, d)
    y = A @ xsparse + std * np.random.randn(n)

    return xsparse, A, y

def f(A, x, y):
    return np.mean(np.square(A@x-y)) / 2

def g(x, lamda):
    return lamda * np.linalg.norm(x, 1)

def F(A, x, y, lamda):
    return f(A, x, y) + g(x, lamda)

@ np.vectorize
def prox_l1(t, gamma):
    return t-gamma if t>gamma else (t+gamma if t<-gamma else 0)

def psga(A, y, lamda, n_iters):

    '''
    Proximal Stochastic Gradient algorithm.

    Parameters
    ----------
    A: matrix of shape (N, D)
    y: vector of shape (N,)
    lamda: positive regularization parameter
    n_iters: number of iterations

    Returns
    -------
    x: vector of shape (D,) estimating the minimizer of the objective
    Fs: array containing the objective function values of the iterates
    em_Fs: array containing the objective function values of the ergodic means
    '''

    # define gamma_0
    gamma_0 = A.shape[0] / np.linalg.norm(A, 'fro')**2
    # sample a sequence of iid random variables on {1, ..., N} for stochastic updates
    i_ks = np.random.choice(A.shape[0], n_iters)
    # initialize the estimate for the minimizer
    x = np.zeros(A.shape[1])
    # initialize book keeping with the object function value of the initial estimate
    Fs = list(); Fs.append(F(A, x, y, lamda))
    # calculate the numerator and the denominator to calculate the ergodic means
    em_num, em_den = gamma_0*x, gamma_0
    # initialize book keeping with the object function value of the 0-th ergodic mean
    em_Fs = list(); em_Fs.append(F(A, em_num/em_den, y, lamda))

    for k, i_k in tqdm(enumerate(i_ks, 1), total=n_iters):
        # compute gamma_k
        gamma_k = gamma_0 / k**0.5
        # compute the argument for the proximal operator
        t = x - gamma_k * (A[i_k]@x - y[i_k]) * A[i_k]
        # update the estimate for the minimizer
        x = prox_l1(t, gamma_k*lamda)
        # update the logs with the objective function value of the current estimate
        Fs.append(F(A, x, y, lamda))
        # update the numerator and the denominator of the ergodic mean
        em_num, em_den = em_num+gamma_k*x, em_den+gamma_k
        # update the logs with the objective function value of the k-th ergodic mean
        em_Fs.append(F(A, em_num/em_den, y, lamda))

    return x, Fs, em_Fs

def rcpga(A, y, lamda, n_iters):

    '''
    Randomized Coordinate Proximal Gradient algorithm.

    Parameters
    ----------
    A: matrix of shape (N, D)
    y: vector of shape (N,)
    lamda: positive regularization parameter
    n_iters: number of iterations

    Returns
    -------
    x: vector of shape (D,) estimating the minimizer of the objective
    Fs: array containing the objective function values of the iterates
    '''

    # define gamma_j for j in {1, ..., D}
    gamma_js = A.shape[0] / np.linalg.norm(A, 2, 0)**2
    # sample a sequence of iid random variables on {1, ..., D} for stochastic updates
    j_ks = np.random.choice(A.shape[1], n_iters)
    # initialize the estimate for the minimizer
    x = np.zeros(A.shape[1])
    # initialize book keeping with the object function value of the initial estimate
    Fs = list(); Fs.append(F(A, x, y, lamda))

    for j_k in tqdm(j_ks, total=n_iters):
        # retrieve gamma_j
        gamma_j = gamma_js[j_k]
        # compute the argument for the proximal operator
        t = x[j_k] - gamma_j/A.shape[0] * A[:, j_k] @ (A@x-y)
        # update the j_k-th coordinate of the estimate for the minimizer
        x[j_k] = prox_l1(t, gamma_j*lamda)
        # update the logs with the objective function value of the current estimate
        Fs.append(F(A, x, y, lamda))

    return x, Fs


n, d, s, std = 1000, 500, 50, 0.06
n_iters, lamda = 100_000, 0.01
x_true, A, y = generate_problem(n=n, d=d, s=s, std=std)

fig, axs = plt.subplots(1, 1, figsize=(12, 6))

x_pred, Fs, em_Fs = psga(A, y, lamda=lamda, n_iters=n_iters)
axs.plot(Fs, label=r'PSGA: $F(x^{k})$', color='blue')
axs.plot(em_Fs, label=r'PSGA: $F(\bar{x}^{k})$', color='green')

x_pred, Fs = rcpga(A, y, lamda=lamda, n_iters=n_iters)
axs.plot(Fs, label=r'RCPGA: $F(x^{k})$', color='firebrick')

axs.set_xlabel(r'Iteration, $k$', size=16)
axs.yaxis.get_major_locator().set_params(integer=True)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.tick_params(axis='both', which='minor', labelsize=12)
axs.grid()
axs.legend(prop={'size':12,'weight':'bold'})
fig.tight_layout()
plt.show()