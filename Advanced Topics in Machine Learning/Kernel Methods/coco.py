import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=100, sigma=0.01):

    t = np.random.uniform(low=0, high=2*np.pi, size=n)
    x = np.sin(t) + np.random.normal(loc=0, scale=sigma, size=n)
    y = np.cos(t) + np.random.normal(loc=0, scale=sigma, size=n)

    return x, y


def gaussian_kernel(x, y=None, beta=1):

    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    if y is None:
        y = x.copy()
    elif y.ndim == 1:
        y = np.expand_dims(y, axis=1)
    assert x.ndim == 2

    difference = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
    norm = np.linalg.norm(difference, axis=2)
    kernel_matrix = np.exp(-beta*norm**2)

    return kernel_matrix


class IncompleteCholesky:

    def compute_decomposition(self, K, eta=0.):

        ell = K.shape[0]
        R = np.zeros((ell, ell))
        d = np.diag(K).copy()

        nu = np.zeros(ell)
        j = 0
        I = np.zeros(ell, dtype=int)
        a = np.max(d)

        while a >= eta and j < ell:
            I[j] = np.argmax(d); nu[j] = np.sqrt(a)
            R[j, :] = (K[I[j], :] - R.T @ R[:, I[j]]) / nu[j]
            d = d - R[j, :] ** 2
            j += 1; a = np.max(d)

        self.T, self.R, self.nu, self.I = j, R[:j, :], nu, I

        return self.R

    def compute_new_features(self, k):

        r = np.zeros(self.T)
        for j in range(self.T):
            r[j] = (k[self.I[j]] - np.dot(r, self.R[:, self.I[j]])) / self.nu[j]
        
        return r
    

def plot(x, y, color='black',
         xticks=None, yticks=None,
         xlabel=None, xlabel_color='black',
         ylabel=None, ylabel_color='black',
         title=None, title_color='black',
         label_size=16, title_size=16, ticks_size=12,
         label_weight='normal', title_weight='normal',
         x_margin=0.1, y_margin=0.1,
         save_name=None):

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, color=color)
    if xticks is not None:
        plt.xlim(min(np.min(x), np.min(xticks))-x_margin, max(np.max(x), np.max(xticks))+x_margin)
        plt.xticks(xticks, fontsize=ticks_size)
    if yticks is not None:
        plt.ylim(min(np.min(y), np.min(yticks))-y_margin, max(np.max(y), np.max(yticks))+y_margin)
        plt.yticks(yticks, fontsize=ticks_size)
    if xlabel is not None:
        plt.xlabel(xlabel, color=xlabel_color, fontsize=label_size, weight=label_weight)
    if ylabel is not None:
        plt.ylabel(ylabel, color=ylabel_color, fontsize=label_size, weight=label_weight)
    if title is not None:
        plt.title(title, color=title_color, fontsize=title_size, weight=title_weight)
    fig.tight_layout()
    plt.grid()
    if save_name is not None:
        plt.savefig(save_name)
        plt.close(fig)
    else:
        plt.show()
    

if __name__ == '__main__':

    from scipy.linalg import eigh
    from scipy.stats import pearsonr

    n, sigma = 1000, 0.01
    beta = 1
    eta = 0.01

    X, Y = generate_data(n, sigma)
    H = np.eye(n) - np.ones((n, n)) / n
    K = gaussian_kernel(X, None, beta); K_tilde = H @ K @ H
    L = gaussian_kernel(Y, None, beta); L_tilde = H @ L @ H

    icholesky = IncompleteCholesky()
    R = icholesky.compute_decomposition(K_tilde, eta); r = R.shape[0]; I_X = icholesky.I[:r]
    S = icholesky.compute_decomposition(L_tilde, eta); s = S.shape[0]; I_Y = icholesky.I[:s]
    
    A = np.block([
        [np.zeros((r, r)), R @ S.T],
        [S @ R.T, np.zeros((s, s))]
    ])
    gamma, coefs = eigh(A, subset_by_index=(r+s-1,r+s-1))

    alpha_tilde = coefs[:r]    # alpha corresponds to K_tilde, which is formed using representations (arranged in R)
    beta_tilde = coefs[-s:]    # beta  corresponds to L_tilde, which is formed using representations (arranged in S)
    f = (R.T @ alpha_tilde).ravel()    # alpha_tilde = R @ alpha, so R.T @ alpha_tilde = K_tilde @ alpha
    g = (S.T @ beta_tilde).ravel()     # beta_tilde  = S @ beta,  so S.T @ beta_tilde  = L_tilde @ beta
    
    plot(
        X, Y, color='black',
        xticks=(-1, -0.5, 0, 0.5, 1), yticks=(-1, -0.5, 0, 0.5, 1),
        xlabel=r'$X$', xlabel_color='darkblue',
        ylabel=r'$Y$', ylabel_color='darkred',
        title=f'Correlation: {pearsonr(X, Y).statistic:.2f}', title_color='black',
        label_size=16, title_size=16, ticks_size=12,
        label_weight='normal', title_weight='bold',
        x_margin=0.1, y_margin=0.1,
        save_name='assets/coco/data.png'
    )

    plot(
        X, f, color='darkblue',
        xticks=(-1, -0.5, 0, 0.5, 1), yticks=(-0.4, -0.2, 0, 0.2, 0.4),
        xlabel=r'$X$', xlabel_color='darkblue',
        title=r'$f(X)$ witness', title_color='darkblue',
        label_size=16, title_size=16, ticks_size=12,
        label_weight='normal', title_weight='normal',
        x_margin=0.1, y_margin=0.05,
        save_name='assets/coco/fx.png'
    )

    plot(
        Y, g, color='darkred',
        xticks=(-1, -0.5, 0, 0.5, 1), yticks=(-0.4, -0.2, 0, 0.2, 0.4),
        xlabel=r'$Y$', xlabel_color='darkred',
        title=r'$g(Y)$ witness', title_color='darkred',
        label_size=16, title_size=16, ticks_size=12,
        label_weight='normal', title_weight='normal',
        x_margin=0.1, y_margin=0.05,
        save_name='assets/coco/gy.png'
    )

    plot(
        f, g, color='black',
        xticks=(-0.4, -0.2, 0, 0.2, 0.4), yticks=(-0.4, -0.2, 0, 0.2, 0.4),
        xlabel=r'$f(X)$', xlabel_color='darkblue',
        ylabel=r'$g(Y)$', ylabel_color='darkred',
        title=f'Pearson Correlation: {pearsonr(f, g).statistic:.2f}\nConstrained Covariance: {gamma[0]/n:.2f}', title_color='black',
        label_size=16, title_size=16, ticks_size=12,
        label_weight='normal', title_weight='bold',
        x_margin=0.05, y_margin=0.05,
        save_name='assets/coco/witness-fns.png'
    )