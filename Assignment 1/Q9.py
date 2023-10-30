import numpy as np
import matplotlib.pyplot as plt

d = 1
n, m = 10, 100

x_train = np.linspace(1, 10, n).reshape(d, n)  # d*n matrix
x_test = np.linspace(1, 10, m).reshape(d, m)   # d*m matrix

true_fn = lambda x: x**2 / 100
y_train = true_fn(x_train).reshape(1, n)
y_noisy = y_train + np.random.normal(loc=0., scale=.1, size=y_train.size).reshape(1, n)
y_test = true_fn(x_test).reshape(1, m)

n_rows, n_cols = 3, 3
cs = (0, 0.01, 0.1, 1, 2, 10, 20, 50, 100); assert len(cs) == n_rows*n_cols

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 20))

for i, c in enumerate(cs, 0):

    x_train_mapped = np.concatenate((x_train, np.sqrt(c)*np.ones(n).reshape(1, n)), axis=0)   # (d+1)*n matrix
    x_test_mapped = np.concatenate((x_test, np.sqrt(c)*np.ones(m).reshape(1, m)), axis=0)     # (d+1)*m matrix

    kernel = lambda a, b: np.dot(a, b) + c
    k_train = np.array([[kernel(a, b) for a in x_train.T] for b in x_train.T])  # n*n matrix
    k_test = np.array([[kernel(a, b) for a in x_train.T] for b in x_test.T])    # m*n matrix

    coefs = x_train_mapped @ np.linalg.inv(k_train+1e-6*np.eye(n)) @ y_noisy.T  # (d+1)*1 matrix
    preds = x_test_mapped.T @ coefs

    axs[i//n_rows, i%n_cols].plot(x_test.ravel(), y_test.ravel(), label='true fn', color='blue')
    axs[i//n_rows, i%n_cols].scatter(x_train.ravel(), y_noisy.ravel(), label='train set', color='green', s=10)
    axs[i//n_rows, i%n_cols].plot(x_test.ravel(), preds.ravel(), label='predictions', color='lightblue')
    axs[i//n_rows, i%n_cols].set_title(f'c = {c}: coefs = {tuple((round(x, 3) for x in coefs.ravel()))}')
    axs[i//n_rows, i%n_cols].legend()
    axs[i//n_rows, i%n_cols].grid()
    
plt.show()

# fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 20))

# for i, c in enumerate(cs, 0):

#     kernel = lambda a, b: np.dot(a, b) + c
#     k_train = np.array([[kernel(a, b) for a in x_train.T] for b in x_train.T])  # n*n matrix
#     k_test = np.array([[kernel(a, b) for a in x_test.T] for b in x_train.T])    # n*m matrix

#     coefs = k_train @ np.linalg.inv(k_train.T@k_train+1e-6*np.eye(n)) @ y_noisy.T  # n*1 matrix
#     # print(f"c = {str(c).ljust(4, ' ')}: coefs = {np.round(coefs.ravel(), 6)}")
#     preds = k_test.T @ coefs

#     axs[i//n_rows, i%n_cols].plot(x_test.ravel(), y_test.ravel(), label='true fn', color='blue')
#     axs[i//n_rows, i%n_cols].scatter(x_train.ravel(), y_noisy.ravel(), label='train set', color='green', s=10)
#     axs[i//n_rows, i%n_cols].plot(x_test.ravel(), preds.ravel(), label='predictions', color='lightblue')
#     axs[i//n_rows, i%n_cols].set_title(f'c = {c}')
#     axs[i//n_rows, i%n_cols].legend()
#     axs[i//n_rows, i%n_cols].grid()
    
# plt.show()