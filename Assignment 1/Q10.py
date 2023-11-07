import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN


train_n, test_n, dim, beta = 10, 400, 2, 1.

def kernel(x1, x2, beta=1.):
    return np.exp(-beta*np.linalg.norm(x1-x2, ord=2, axis=-1)**dim)


linspace = np.linspace(1., 10., num=train_n).reshape(-1, 1)
axes = np.meshgrid(*[linspace for _ in range(dim)])
train_x = np.concatenate([axis.reshape(-1, 1) for axis in axes], axis=1)
train_x = train_x[np.random.choice(np.arange(train_x.shape[0]), replace=False, size=50), :]
train_y = np.random.choice((-1, 1), size=train_x.shape[0]).reshape(-1, 1)

train_k = kernel(np.expand_dims(train_x, 0), np.expand_dims(train_x, 1), beta=beta)
weights = np.linalg.inv(train_k) @ train_y

linspace = np.linspace(1., 10., num=test_n).reshape(-1, 1)
axes = np.meshgrid(*[linspace for _ in range(dim)])
test_x = np.concatenate([axis.reshape(-1, 1) for axis in axes], axis=1)

test_k = kernel(np.expand_dims(test_x, 0), np.expand_dims(train_x, 1), beta=beta)
preds = test_k.T @ weights


fig, axs = plt.subplots(2, 1, figsize=(12, 16))
axs[0].scatter(test_x[(preds<0).ravel(), 0], test_x[(preds<0).ravel(), 1], color='lightcoral', s=1)
axs[0].scatter(test_x[(preds>0).ravel(), 0], test_x[(preds>0).ravel(), 1], color='lightgreen', s=1)
axs[0].scatter(train_x[(train_y<0).ravel(), 0], train_x[(train_y<0).ravel(), 1], color='red', s=20)
axs[0].scatter(train_x[(train_y>0).ravel(), 0], train_x[(train_y>0).ravel(), 1], color='green', s=20)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title(fr'Gaussian Kernel Regression $\beta = {beta}$')


knn = KNN(n_neighbors=1, algorithm='brute')
knn.fit(train_x, train_y.ravel())
preds = knn.predict(test_x)

axs[1].scatter(test_x[(preds<0).ravel(), 0], test_x[(preds<0).ravel(), 1], color='lightcoral', s=1)
axs[1].scatter(test_x[(preds>0).ravel(), 0], test_x[(preds>0).ravel(), 1], color='lightgreen', s=1)
axs[1].scatter(train_x[(train_y<0).ravel(), 0], train_x[(train_y<0).ravel(), 1], color='red', s=20)
axs[1].scatter(train_x[(train_y>0).ravel(), 0], train_x[(train_y>0).ravel(), 1], color='green', s=20)
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title(r'k-Nearest Neighbors with $k = 1$')

plt.savefig(f'assets/q10_kernel-reg-vs-knn_beta-{int(beta)}.png')
plt.show()