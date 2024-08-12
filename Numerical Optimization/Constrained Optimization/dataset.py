import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

N, d, r = 300, 2, 1.4
X, Y = datasets.make_circles(n_samples=N, shuffle=True, noise=0.08, random_state=42, factor=0.5)
X, Y = 2*X, 2*Y-1

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
train_S = np.hstack((train_X, train_Y[:, None]))
test_S = np.hstack((test_X, test_Y[:, None]))
S = np.vstack((train_S, test_S))

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0][Y == +1.], X[:, 1][Y == +1.], s=20, color='darkgreen')
    plt.scatter(X[:, 0][Y == -1.], X[:, 1][Y == -1.], s=20, color='darkblue')
    plt.tight_layout()
    plt.grid()
    plt.savefig('assets/2-circles.png')
    plt.show()