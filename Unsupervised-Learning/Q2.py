import numpy as np
from scipy.special import beta

# Load the dataset
dataset = np.loadtxt('binarydigits.txt')
# We have N=100 images, each with D=64 pixels laid out as a row vector
N, D = dataset.shape

# Relative probabilities
model_a = 2**(-N*D)
model_b = beta(1+np.sum(dataset), 1+N*D-np.sum(dataset))
model_c = np.prod(
    beta(1+np.sum(dataset, axis=0), 
    1+N-np.sum(dataset, axis=0))
)

print(model_a, model_b, model_c)    
# 0.0 0.0 0.0

def log_fact(n, base=2):
    '''
    Calculate the log of the factorial of a positive integer n.
    :param n: positive integer n for which the log(n!) will be returned. 
    :return:
    log_fact_n: calculated as log(n!) = \sum_{i=1}^n log(i)
    '''
    log_fact_n = np.sum(np.log(np.arange(1, n+1))) / np.log(base)
    return log_fact_n

# Since the relative probabilities evaluate to 0, we work with their log
log_model_a = -N*D
log_model_b = log_fact(np.sum(dataset)) + \
    log_fact(N*D-np.sum(dataset)) - \
    log_fact(1+N*D)
log_model_c = np.sum([log_fact(np.sum(dataset[:, d])) + \
    log_fact(N-np.sum(dataset[:, d])) - \
    log_fact(1+N) for d in range(D)])
    
print(log_model_a, log_model_b, log_model_c)    
# -6400 -6180.103537486546 -5556.101001247703