import numpy as np
import matplotlib.pyplot as plt


class MarkovChain:

    def __init__(self, source, symbols):

        self.source = source
        self.symbols = symbols
        self.symbols_to_idx = {s: i for i, s in enumerate(symbols)}
        
        # Laplace smoothing: setting uniform prior, avoids transition probabilities 0
        transition_counts = np.ones((len(symbols), len(symbols)))
        # Fill the transition matrix with transition counts
        for beta, alpha in zip(source[:-1], source[1:]):
            transition_counts[self.symbols_to_idx[beta], self.symbols_to_idx[alpha]] += 1
        # Normalize the counts to get the transition probabilities
        self.tm = transition_counts / transition_counts.sum(axis=1, keepdims=True)

        # Calculate the eigenvalue decomposition of the transition matrix
        eig_vals, eig_vecs = np.linalg.eig(self.tm.T)
        # Find the invariant distribution
        p_inv = eig_vecs[:, np.where(np.isclose(eig_vals, 1))[0][0]]
        # Normalize the invariant distribution to sum to 1
        self.p_inv = (p_inv / np.sum(p_inv)).astype(float)

    def transition_prob(self, beta, alpha):
        '''
        Return the probability of transitioning from \beta to \alpha, P(s_{i+1}=\alpha|s_i=\beta).
        :param beta: the first character in the transition.
        :param alpha: the second character in the transition.
        :returns:
        tp: the transition probability, P(s_{i+1}=\alpha|s_i=\beta).
        '''
        tp = self.tm[self.symbols_to_idx[beta], self.symbols_to_idx[alpha]]
        return tp

    def limiting_prob(self, gamma):
        '''
        Calculating the marginal probability for a character, \gamma.
        :param gamma: character  for which we return the marginal probability.
        :returns:
        mp: marginal probability of observing gamma.
        '''
        mp = self.p_inv[self.symbols_to_idx[gamma]]
        return mp
    

# Create a list of symbols in our dictionary.
with open('symbols.txt', 'r', encoding='utf-16') as f:
    symbols = f.read().split('\n')

# Read War and Peace.
with open('war-and-peace.txt', 'r', encoding='utf-8') as f:
    war_and_peace = f.read()
    war_and_peace = ' '.join(war_and_peace.lower().split())

# Filter out symbols in War and Peace that are not in our dictionary.
unrecognized_symbols = set(war_and_peace).difference(symbols)
for uc in unrecognized_symbols:
    war_and_peace = war_and_peace.replace(uc, ' ')
war_and_peace = ' '.join(war_and_peace.split())

# Create a Markov Chain.
mc = MarkovChain(war_and_peace, symbols)


if __name__ == '__main__':

    fig, axs = plt.subplots(figsize=(10, 10))
    axs.imshow(mc.tm, cmap='gray_r')
    axs.set_xticks(np.arange(53)); axs.set_xticklabels(symbols, size=12); axs.xaxis.tick_top()
    axs.set_yticks(np.arange(53)); axs.set_yticklabels(symbols, size=12)
    axs.set_ylabel('First symbol', size=16)
    axs.set_title('Second symbol', size=16)
    fig.tight_layout()
    plt.savefig('assets/mcmc/tm.png')
    plt.close(fig)

    fig, axs = plt.subplots(figsize=(10, 1))
    axs.imshow(mc.p_inv.reshape(1, -1), cmap='gray_r')
    axs.set_xticks(np.arange(53)); axs.set_xticklabels(symbols, size=12)
    axs.set_yticks([])
    fig.tight_layout()
    plt.savefig('assets/mcmc/lim-prob.png')
    plt.close(fig)