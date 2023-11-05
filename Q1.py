import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = np.loadtxt('binarydigits.txt')
# We have N=100 images, each with D=64 pixels laid out as a row vector
N, D = dataset.shape

def plot(estimates, fn):
    '''
    Plots the estimates as black-and-white image.
    :param estimates: the parameter estimates (MLE or MAP).
    '''
    fig, axs = plt.subplots(figsize=(4, 4))
    estimates = estimates.reshape(8, 8)
    axs.imshow(estimates, cmap='gray')
    axs.set_xticks(np.arange(8)); axs.set_yticks(np.arange(8))
    for n in range(64):
        i, j = n//8, n%8
        axs.text(j, i, f'{estimates[i, j]:.2f}', ha='center', 
            va='center', color='orangered', weight='bold', size=10)
    fig.tight_layout()
    plt.savefig(f'assets/estimates/{fn}')
    plt.close(fig)

# The MLE estimate is given as 
# \hat{p}_{MLE} = \frac{1}{N} \sum_{i=1}^N x^{(n)}
ml_estimate = np.mean(dataset, axis=0)
# Plot the MLE estimate as an image 
plot(ml_estimate, fn='mle.png')

# Parameters of the Beta prior.
alpha = beta = 3
# The MAP estimate is given as 
# \hat{p}_{MAP} = \frac{\alpha-1+\sum_{i=1}^N x^{(n)}}{\alpha+\beta-2+N}
map_estimate = (alpha-1+np.sum(dataset, axis=0)) / (alpha+beta-2+N)
# Plot the MAP estimate as an image 
plot(map_estimate, fn='map.png')