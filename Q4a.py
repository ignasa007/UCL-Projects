import numpy as np
import matplotlib.pyplot as plt
from ssm_kalman import run_ssm_kalman

X = np.loadtxt('ssm_spins.txt').T

# Randomly generate the initial state, y_1 ~ N(0, I)
Q_init = np.identity(4)
y_init = np.random.multivariate_normal(np.zeros(4), Q_init)

# Define the parameters of the LGSSM model
A = 0.99 * np.array([
    [np.cos(2*np.pi/180), -np.sin(2*np.pi/180), 0, 0],
    [np.sin(2*np.pi/180), np.cos(2*np.pi/180), 0, 0],
    [0, 0, np.cos(2*np.pi/90), -np.sin(2*np.pi/90)],
    [0, 0, np.sin(2*np.pi/90), np.cos(2*np.pi/90)],
])
Q = np.identity(4) - A@A.T
C = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [0.5, 0.5, 0.5, 0.5]
])
R = np.identity(5)

# Run Kalman filtering and Kalman smoothing with the randomly generated initial latent state
Y_filt, V_filt, _, L_filt = run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='filt')
Y_smooth, V_smooth, Vj_smooth, L_smooth = run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='smooth')

# Plot the expectations, Y_t, of the latents, y_t
for mode, Y in zip(('filter', 'smooth'), (Y_filt, Y_smooth)):
    fig, axs = plt.subplots(1, 1)
    for k, Y_k in enumerate(Y):
        axs.plot(Y_k, color=f'C{k}', label=f'$k = {k+1}$')
    axs.set_xlabel(r'$t$')
    axs.set_ylabel(r'$\mathbb{{E}}[(y_t)_k]$')
    axs.grid()
    axs.legend()
    fig.tight_layout()
    plt.savefig(f'assets/kalman/y_{mode}.png')
    plt.close(fig)

# Plot log(det(V_t)) for the covariances, V_t, of the latents, y_t
for mode, V in zip(('filter', 'smooth'), (V_filt, V_smooth)):
    logdet = [2*np.sum(np.log(np.diag(v))) for v in np.linalg.cholesky(V)]
    fig, axs = plt.subplots(1, 1)
    axs.plot(logdet, color='green')
    axs.set_xlabel(r'$t$')
    axs.grid()
    fig.tight_layout()
    plt.savefig(f'assets/kalman/v_{mode}.png')
    plt.close(fig)

# Plot the components, k, of Y_filt and Y_smooth to demonstrate 
# the "smoothing" effect of Kalman smoothing
for comp in range(4):
    fig, axs = plt.subplots(1, 1)
    axs.plot(Y_filt[comp], color='royalblue', label='Filtering')
    axs.plot(Y_smooth[comp], color='black', label='Smoothing')
    axs.set_xlabel(r'$t$')
    axs.set_ylabel(fr'$\mathbb{{E}}[(y_t)_{comp+1}]$')
    axs.grid()
    axs.legend()
    fig.tight_layout()
    plt.savefig(f'assets/kalman/comp_{comp+1}.png')
    plt.close(fig)