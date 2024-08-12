import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
from Q6a import read_data, GibbsSampler, GibbsSamplerCollapsed

docs_words_train, docs_words_test = read_data('./toyexample.data')
n_docs, n_words = docs_words_train.shape
n_topics = 3
random_seed = 0

grid = (0.5, 0.9, 1, 3, 5)
logjoint = [[None for _ in grid] for _ in grid]
logpred = [[None for _ in grid] for _ in grid]

for i, alpha in enumerate(grid):
    for j, beta in enumerate(grid):
        sampler = GibbsSamplerCollapsed(n_docs=n_docs, n_topics=n_topics, n_words=n_words, alpha=alpha, beta=beta, random_seed=random_seed)
        sampler.run(docs_words_train, docs_words_test, n_iter=200, save_loglike=True)
        like_train, like_test = sampler.get_loglike()
        logjoint[i][j] = np.mean(like_train[-10:])
        logpred[i][j] = np.mean(like_test[-10:])
logjoint = pd.DataFrame(data=logjoint, index=grid, columns=grid)
logpred = pd.DataFrame(data=logpred, index=grid, columns=grid)

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
sns.heatmap(logjoint, xticklabels=grid, yticklabels=grid, ax=axs[0])
axs[0].set_xlabel(r'$\beta$', size=22)
axs[0].set_ylabel(r'$\alpha$', size=22)
axs[0].set_title('Log-joint', size=22)
sns.heatmap(logpred, xticklabels=grid, yticklabels=grid, ax=axs[1])
axs[1].set_xlabel(r'$\beta$', size=22)
axs[1].set_ylabel(r'$\alpha$', size=22)
axs[1].set_title('Log-predictive', size=22)
fig.tight_layout()
plt.savefig('assets/lda/col-hparams-toy.png')
plt.close(fig)
plt.show()