import numpy as np
from Q2_1 import generate_data, gaussian_kernel, IncompleteCholesky, plot


if __name__ == '__main__':

    from scipy.linalg import eigh
    from scipy.stats import pearsonr

    n, sigma = 1000, 0.01
    beta = 1
    n_eps = 10

    X, Y = generate_data(n, sigma)
    H = np.eye(n) - np.ones((n, n)) / n
    K = gaussian_kernel(X, None, beta); K_tilde = H @ K @ H
    L = gaussian_kernel(Y, None, beta); L_tilde = H @ L @ H

    icholesky = IncompleteCholesky()
    R = icholesky.compute_decomposition(K_tilde); r = R.shape[0]; I_X = icholesky.I[:r]
    S = icholesky.compute_decomposition(L_tilde); s = S.shape[0]; I_Y = icholesky.I[:s]
    
    A = np.block([
        [np.zeros((r, r)), R @ S.T],
        [S @ R.T, np.zeros((s, s))]
    ])
    B = np.block([
        [R @ R.T + n_eps*np.eye(r), np.zeros((r, s))],
        [np.zeros((s, r)), S @ S.T + n_eps*np.eye(s)]
    ])
    gamma, coefs = eigh(A, B, subset_by_index=(r+s-1,r+s-1))

    alpha = coefs[:r]; f = (R.T @ alpha).ravel()
    beta = coefs[-s:]; g = (S.T @ beta).ravel()

    plot(
        X, Y, color='black',
        xticks=(-1, -0.5, 0, 0.5, 1), yticks=(-1, -0.5, 0, 0.5, 1),
        xlabel=r'$X$', xlabel_color='darkblue',
        ylabel=r'$Y$', ylabel_color='darkred',
        title=f'Correlation: {pearsonr(X, Y).statistic:.2f}', title_color='black',
        label_size=16, title_size=16, ticks_size=12,
        label_weight='normal', title_weight='bold',
        x_margin=0.1, y_margin=0.1,
        save_name='assets/cca/data.png'
    )

    plot(
        X, f, color='darkblue',
        xticks=(-1, -0.5, 0, 0.5, 1), yticks=(-0.04, -0.02, 0, 0.02, 0.04),
        xlabel=r'$X$', xlabel_color='darkblue',
        title=r'$f(X)$ witness', title_color='darkblue',
        label_size=16, title_size=16, ticks_size=12,
        label_weight='normal', title_weight='normal',
        x_margin=0.1, y_margin=0.005,
        save_name='assets/cca/fx.png'
    )

    plot(
        Y, g, color='darkred',
        xticks=(-1, -0.5, 0, 0.5, 1), yticks=(-0.04, -0.02, 0, 0.02, 0.04),
        xlabel=r'$Y$', xlabel_color='darkred',
        title=r'$g(Y)$ witness', title_color='darkred',
        label_size=16, title_size=16, ticks_size=12,
        label_weight='normal', title_weight='normal',
        x_margin=0.1, y_margin=0.005,
        save_name='assets/cca/gy.png'
    )

    plot(
        f, g, color='black',
        xticks=(-0.04, -0.02, 0, 0.02, 0.04), yticks=(-0.04, -0.02, 0, 0.02, 0.04),
        xlabel=r'$f(X)$', xlabel_color='darkblue',
        ylabel=r'$g(Y)$', ylabel_color='darkred',
        title=f'Pearson Correlation: {pearsonr(f, g).statistic:.2f}\nCanonical Correlation: {gamma[0]:.2f}', title_color='black',
        label_size=16, title_size=16, ticks_size=12,
        label_weight='normal', title_weight='bold',
        x_margin=0.005, y_margin=0.005,
        save_name='assets/cca/witness-fns.png'
    )