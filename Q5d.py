from collections import Counter
import numpy as np
from Q5a import mc


with open('message.txt', 'r', encoding='utf-16') as f:
    message = f.read()

def log_likelihood(sigma):
    '''
    Calculate the log-likelihood of observing the message under a given permutation.
    :param sigma: the encoding rule, defined as a permutation of the symbols.
    :returns:
    log_likelihood: log-likelihood of observing the data under the encoding, \sigma.
    '''

    # Compute the decoding from the given encoding.
    sigma_inv = {e: d for d, e in sigma.items()}
    # Decrypt the text using the deconding.
    decrypted_text = ''.join([sigma_inv[e] for e in message])

    # Calculate the log-likelihood of the decrypted message using the Markov chain.
    log_likelihood = np.log(mc.limiting_prob(decrypted_text[0]))
    for beta, alpha in zip(decrypted_text[:-1], decrypted_text[1:]):
        transition_prob = mc.transition_prob(beta, alpha)
        log_likelihood += np.log(transition_prob+1e-10)
        
    return log_likelihood

# Make counters for the symbols in War and Peace and in the message.
war_and_peace_cntr = dict(Counter(mc.source))
message_cntr = Counter(message)
message_cntr = {k: message_cntr.get(k, 0) for k in war_and_peace_cntr}

# Intelligent initialization of the Markov chain: encode the k-th most common 
# symbol in War and Peace as the k-th most common symbol in the message.
permutation = {
    d: e for d, e in zip(
        sorted(war_and_peace_cntr.keys(), key=lambda x: war_and_peace_cntr[x]), 
        sorted(message_cntr.keys(), key=lambda x: message_cntr[x])
    )
}

# Run MCMC for 15000 iterations.
from tqdm import tqdm
for i in tqdm(range(1, 15001)):
    # Sample two different symbols for which the encoding will be swapped.
    s, s_ = np.random.choice(mc.symbols, 2, replace=False)
    # Define the proposal by swapping the encoding of the sampled symbols in the current permutation.
    proposal = permutation.copy()
    proposal[s], proposal[s_] = proposal[s_], proposal[s]
    # Calculate the increment in likelihood.
    log_likelihood_diff = log_likelihood(proposal) - log_likelihood(permutation)
    # In part (c), we saw that S(s->s_) = S(s_->s), so we check if the likelihood ratio > Z~U[0,1].
    if np.random.uniform() <= np.exp(log_likelihood_diff):
        permutation = proposal.copy()
    # Note the decoding of the first 60 characters of the message after every 100 iterations. 
    if i%100 == 0:
        permutation_inv = {e: d for d, e in permutation.items()}
        with open(f'assets/mcmc/decrypted.txt', 'a') as f:
            f.write(''.join([permutation_inv[e] for e in message[:60]])+'\n')

# Print the decoded message.
print(''.join([permutation_inv[e] for e in message]))