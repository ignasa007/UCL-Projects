import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
from Q6a import read_data, GibbsSampler, GibbsSamplerCollapsed

docs_words_train, docs_words_test = read_data('./toyexample.data')
n_docs, n_words = docs_words_train.shape
random_seed = 0
alpha = beta = 1
ks = (2, 3, 4, 5, 6)

std_logjoint, std_logpred = [None for _ in ks], [None for _ in ks]
for i, n_topics in enumerate(ks):
    sampler = GibbsSampler(n_docs=n_docs, n_topics=n_topics, n_words=n_words, alpha=alpha, beta=beta, random_seed=random_seed)
    sampler.run(docs_words_train, docs_words_test, n_iter=300, save_loglike=True)
    like_train, like_test = sampler.get_loglike()
    std_logjoint[i] = np.mean(like_train[-10:])
    std_logpred[i] = np.mean(like_test[-10:])

col_logjoint, col_logpred = [None for _ in ks], [None for _ in ks]
for i, n_topics in enumerate(ks):
    sampler = GibbsSamplerCollapsed(n_docs=n_docs, n_topics=n_topics, n_words=n_words, alpha=alpha, beta=beta, random_seed=random_seed)
    sampler.run(docs_words_train, docs_words_test, n_iter=300, save_loglike=True)
    like_train, like_test = sampler.get_loglike()
    col_logjoint[i] = np.mean(like_train[-10:])
    col_logpred[i] = np.mean(like_test[-10:])

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

axs[0].plot(ks, std_logjoint, color='blue', label='Standard Gibbs')
axs[0].plot(ks, col_logjoint, color='green', label='Collapsed Gibbs')
axs[0].set_xlabel('Number of Topics', size=22)
axs[0].set_ylabel('Log-joint', size=22)
axs[0].grid()

axs[1].plot(ks, std_logpred, color='blue', label='Standard Gibbs')
axs[1].plot(ks, col_logpred, color='green', label='Collapsed Gibbs')
axs[1].set_xlabel('Number of Topics', size=22)
axs[1].set_ylabel('Log-predictive', size=22)
axs[1].grid()
axs[1].legend()

fig.tight_layout()
plt.savefig('assets/lda/n_topics-toy.png')
plt.close(fig)
plt.show()