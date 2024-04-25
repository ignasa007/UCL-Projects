import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
from statsmodels.graphics.tsaplots import plot_acf
from Q6a import read_data, GibbsSampler, GibbsSamplerCollapsed

print('Running toyexample.data with the standard sampler')

docs_words_train, docs_words_test = read_data('./toyexample.data')
n_docs, n_words = docs_words_train.shape
n_topics = 3
alpha = 1
beta = 1
random_seed = 0
n_iter = 500

sampler = GibbsSampler(
    n_docs=n_docs, n_topics=n_topics, n_words=n_words,
    alpha=alpha, beta=beta, random_seed=random_seed
)

topic_doc_words_distr, theta, phi = sampler.run(
    docs_words_train, docs_words_test,
    n_iter=n_iter, save_loglike=True
)

like_train, like_test = sampler.get_loglike()
like_train, like_test = like_train[20:], like_test[20:]

fig, axs = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(like_train, ax=axs[0])
axs[0].set_ylabel('Auto(log-joint)', size=22)
axs[0].set_title('Train Subset of Toy Data', size=22)
plot_acf(like_test, ax=axs[1])
axs[1].set_ylabel('Auto(log-predictive)', size=22)
axs[1].set_xlabel('L', size=22)
axs[1].set_title('Test Subset of Toy Data', size=22)
fig.tight_layout()
plt.savefig('assets/lda/std-gibbs-autocorr-toy.png')
plt.close(fig)

print('Running toyexample.data with the collapsed sampler')

sampler_collapsed = GibbsSamplerCollapsed(
    n_docs=n_docs, n_topics=n_topics, n_words=n_words,
    alpha=alpha, beta=beta, random_seed=random_seed
)

doc_word_samples = sampler_collapsed.run(
    docs_words_train, docs_words_test,
    n_iter=n_iter, save_loglike=True
)

like_train, like_test = sampler_collapsed.get_loglike()
like_train, like_test = like_train[20:], like_test[20:]

fig, axs = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(like_train, ax=axs[0])
axs[0].set_ylabel('Auto(log-joint)', size=22)
axs[0].set_title('Train Subset of Toy Data', size=22)
plot_acf(like_test, ax=axs[1])
axs[1].set_ylabel('Auto(log-predictive)', size=22)
axs[1].set_xlabel('L', size=22)
axs[1].set_title('Test Subset of Toy Data', size=22)
fig.tight_layout()
plt.savefig('assets/lda/col-gibbs-autocorr-toy.png')
plt.close(fig)