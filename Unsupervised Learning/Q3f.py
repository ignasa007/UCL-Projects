import gzip
import os
import shutil
import numpy as np
from Q3d import EMForMultivariateBernoulli, plot_params

# Load the dataset
dataset = np.loadtxt('binarydigits.txt')

# Loop over the values of K, running EM for each of them
Ks = (2, 3, 4, 7, 10)
compression_length = list()
for K in Ks:
    em = EMForMultivariateBernoulli()
    P, pi, R, log_likelihoods = em(K=K, X=dataset, n_iterations=15, eps=1e-1)
    # Record the theoretical limit on compression length
    compression_length.append(-log_likelihoods[-1]/np.log(2))

print(compression_length)

# Function to compress a file
def compress_file(file_path):
    with open(file_path, 'rb') as f_in:
        with gzip.open(file_path + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Function to calculate file size in bits
def get_file_size_in_bits(file_path):
    return os.path.getsize(file_path) * 8  # Convert bytes to bits

# Replace 'yourfile.txt' with the path to your file
file_path = 'binarydigits.txt'
# Convert to binary
with open(file_path, 'r') as f:
    bits = ''.join(f.read().split())
bytes = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
bin_file_path = os.path.splitext(file_path)[0] + '.bin'
with open(bin_file_path, 'wb') as f:
    f.write(bytes)
# Compress the file
compress_file(bin_file_path)
# Calculate the size of the compressed file in bits
compressed_file_size = get_file_size_in_bits(bin_file_path + '.gz')

print(f"Size of the compressed file in bits: {compressed_file_size}")
# Size of the compressed file in bits: 5552