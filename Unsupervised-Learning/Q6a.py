# -*- coding: utf-8 -*-

"""
    File name: gibbs_sampler.py
    Description: a re-implementation of the Gibbs sampler for http://www.gatsby.ucl.ac.uk/teaching/courses/ml1
    Author: python: Roman Pogodin, MATLAB (original): Yee Whye Teh and Maneesh Sahani
    Date created: October 2018
    Python version: 3.6
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)


# todo: sample everything from self.rang_gen to control the random seed (works as numpy.random)
class GibbsSampler:
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          number of documents
        :param n_topics:        number of topics
        :param n_words:         number of words in vocabulary
        :param alpha:           dirichlet parameter on topic mixing proportions
        :param beta:            dirichlet parameter on topic word distributions
        :param random_seed:     random seed of the sampler
        """
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.n_words = n_words
        self.alpha = alpha
        self.beta = beta
        self.rand_gen = np.random.RandomState(random_seed)

        self.docs_words = np.zeros((self.n_docs, self.n_words))
        self.docs_words_test = None
        self.loglike = None
        self.loglike_test = None
        self.do_test = False

        self.A_dk = np.zeros((self.n_docs, self.n_topics))  # number of words in document d assigned to topic k
        self.B_kw = np.zeros((self.n_topics, self.n_words))  # number of occurrences of word w assigned to topic k
        self.A_dk_test = np.zeros((self.n_docs, self.n_topics))
        self.B_kw_test = np.zeros((self.n_topics, self.n_words))

        self.theta = np.ones((self.n_docs, self.n_topics)) / self.n_topics  # theta[d] is the distribution over topics in document d
        self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words  # phi[k] is the distribution words in topic k

        self.topics_space = np.arange(self.n_topics)
        self.topic_doc_words_distr = np.zeros((self.n_topics, self.n_docs, self.n_words))  # z_id|x_id, theta, phi

    def init_sampling(self, docs_words, docs_words_test=None,
                      theta=None, phi=None, n_iter=0, save_loglike=False):
        
        assert np.all(docs_words.shape == (self.n_docs, self.n_words)), "docs_words shape=%s must be (%d, %d)" % (
            docs_words.shape, self.n_docs, self.n_words)
        
        self.n_docs = docs_words.shape[0]

        self.docs_words = docs_words
        self.docs_words_test = docs_words_test

        self.do_test = (docs_words_test is not None)

        if save_loglike:
            self.loglike = np.zeros(n_iter)

            if self.do_test:
                self.loglike_test = np.zeros(n_iter)

        self.A_dk.fill(0.0)
        self.B_kw.fill(0.0)
        self.A_dk_test.fill(0.0)
        self.B_kw_test.fill(0.0)

        self.init_params(theta, phi)

    def init_params(self, theta=None, phi=None):
        if theta is None:
            self.theta = np.ones((self.n_docs, self.n_topics)) / self.n_topics
        else:
            self.theta = theta.copy()

        if phi is None:
            self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words
        else:
            self.phi = phi.copy()

        self.update_topic_doc_words()
        self.sample_counts()

    def run(self, docs_words, docs_words_test=None,
            n_iter=100, theta=None, phi=None, save_loglike=False):
        """
        docs_words is a matrix n_docs * n_words; each entry
        is a number of occurrences of a word in a document
        docs_words_test does not influence the updates and is used
        for validation
        """
        self.init_sampling(docs_words, docs_words_test,
                           theta, phi, n_iter, save_loglike)

        for iteration in range(n_iter):
            self.update_params()

            if save_loglike:
                self.update_loglike(iteration)

        return self.to_return_from_run()

    def to_return_from_run(self):
        return self.topic_doc_words_distr, self.theta, self.phi

    def update_params(self):
        """
        Samples theta and phi, then computes the distribution of
        z_id and samples counts A_dk, B_kw from it
        """
        # todo: sample theta and phi
        for d in range(self.n_docs):
            self.theta[d] = self.rand_gen.dirichlet(self.A_dk[d]+self.alpha)
        for k in range(self.n_topics):
            self.phi[k] = self.rand_gen.dirichlet(self.B_kw[k]+self.alpha)
        self.update_topic_doc_words()
        self.sample_counts()

    def update_topic_doc_words(self):
        """
        Computes the distribution of z_id|x_id, theta, phi
        """
        self.topic_doc_words_distr = np.repeat(
            self.theta.T[:, :, None], self.n_words, axis=2) * self.phi[:, None, :]
        # shape = (n_topics, n_docs, n_words)
        self.topic_doc_words_distr /= self.theta.dot(self.phi)[None, :, :]

    def sample_counts(self):
        """
        For each document and each word, samples from z_id|x_id, theta, phi
        and adds the results to the counts A_dk and B_kw
        """
        self.A_dk.fill(0)
        self.B_kw.fill(0)
        if self.do_test:
            self.A_dk_test.fill(0)
            self.B_kw_test.fill(0)
        # todo: sample a topic for each (doc, word) and update A_dk, B_kw correspondingly
        for doc in range(self.n_docs):
            for word in range(self.n_words):
                # Sample topic assignments using the distribution over topics for w_id.
                sampled_topics = self.rand_gen.choice(
                    self.topics_space, 
                    size=self.docs_words[doc, word] + (self.docs_words_test[doc, word] if self.do_test else 0),
                    p=self.topic_doc_words_distr[:, doc, word]
                )
                # Update topic assignments for the training set.
                sampled_topics_train = sampled_topics[:self.docs_words[doc, word]]
                sample, counts = np.unique(sampled_topics_train, return_counts=True)
                self.A_dk[doc, sample] += counts
                self.B_kw[sample, word] += counts
                # Update topic assignments for the test set. 
                sampled_topics_test = sampled_topics[-self.docs_words_test[doc, word]:]
                sample, counts = np.unique(sampled_topics_test, return_counts=True)
                self.A_dk_test[doc, sample] += counts
                self.B_kw_test[sample, word] += counts


    def update_loglike(self, iteration):
        """
        Updates loglike of the data, omitting the constant additive term
        with Gamma functions of hyperparameters
        """
        # todo: implement log-like
        # Log of joint probability of the observations and the latents, logp({x},{z},{\phi},{\theta})
        self.loglike[iteration] = (self.A_dk+self.alpha-1).ravel().dot(np.log(self.theta).ravel()) \
            + (self.B_kw+self.beta-1).ravel().dot(np.log(self.phi).ravel())
        # Copied over from the stdgibbs_logpred.m MATLAB file.
        self.loglike_test[iteration] = np.sum(np.log(self.theta@self.phi)*self.docs_words_test)
        
    def get_loglike(self):
        """Returns log-likelihood at each iteration."""
        if self.do_test:
            return self.loglike, self.loglike_test
        else:
            return self.loglike


class GibbsSamplerCollapsed(GibbsSampler):
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          number of documents
        :param n_topics:        number of topics
        :param n_words:         number of words in vocabulary
        :param alpha:           dirichlet parameter on topic mixing proportions
        :param beta:            dirichlet parameter on topic word distributions
        :param random_seed:     random seed of the sampler
        """
        super().__init__(n_docs, n_topics, n_words, alpha, beta, random_seed)

        # topics assigned to each (doc, word)
        self.doc_word_samples = np.ndarray((self.n_docs, self.n_words), dtype=object)
        self.doc_word_samples_test = self.doc_word_samples.copy()

    def init_params(self, theta=None, phi=None):
        # z_id are initialized uniformly
        for doc in range(self.n_docs):
            for word in range(self.n_words):
                if self.do_test:
                    additional_samples = self.docs_words_test[doc, word]
                else:
                    additional_samples = 0

                sampled_topics = self.rand_gen.choice(self.topics_space,
                                                      size=self.docs_words[doc, word] + additional_samples)

                sampled_topics_train = sampled_topics[:self.docs_words[doc, word]]
                self.doc_word_samples[doc, word] = sampled_topics_train.copy()  # now each cell is an np.array

                sample, counts = np.unique(sampled_topics_train, return_counts=True)

                self.A_dk[doc, sample] += counts
                self.B_kw[sample, word] += counts

                if self.do_test:
                    sampled_topics_test = sampled_topics[self.docs_words[doc, word]:]
                    self.doc_word_samples_test[doc, word] = sampled_topics_test.copy()

                    sample, counts = np.unique(sampled_topics_test, return_counts=True)

                    self.A_dk_test[doc, sample] += counts
                    self.B_kw_test[sample, word] += counts

    def update_params(self):
        """
        Computes the distribution of z_id.
        Sampling of A_dk, B_kw is done automatically as
        each new z_id updates these counters
        """
        # todo: sample a topic for each (doc, word) and update A_dk, B_kw correspondingly
        for d in range(self.n_docs):
            for w in range(self.n_words):
                B_k = self.B_kw.sum(axis=1)
                for i, topic in enumerate(self.doc_word_samples[d, w]):
                    # Conditioning on the current word, so remover the topic assignment.
                    self.A_dk[d, topic] -= 1; self.B_kw[topic, w] -= 1; B_k[topic] -= 1
                    # Calculate the sampling probability upto a factor.
                    sampling_weights = (self.A_dk[d, :]+self.alpha) \
                        * (self.B_kw[:, w]+self.beta) / (B_k + self.n_words*self.beta)
                    # Calculate the sampling probabilities.
                    sampling_probs = sampling_weights / sampling_weights.sum()
                    # Sample a random topic based on the sampling distribution above.
                    topic = self.rand_gen.choice(self.topics_space, p=sampling_probs)
                    # Book keeping.
                    self.doc_word_samples[d, w][i] = topic
                    # Update the counters.
                    self.A_dk[d, topic] += 1; self.B_kw[topic, w] += 1; B_k[topic] += 1

    def update_loglike(self, iteration):
        """
        Updates loglike of the data, omitting the constant additive term
        with Gamma functions of hyperparameters
        """
        # todo: implement log-like
        # Log of joint probability of the observations and the latents, logp({x},{z}).
        log_like = 0.0
        for d in range(self.n_docs):
            log_like += np.sum(gammaln(self.A_dk[d] + self.alpha))
            log_like -= gammaln(np.sum(self.A_dk[d] + self.alpha))
        for k in range(self.n_topics):
            log_like += np.sum(gammaln(self.B_kw[k] + self.beta))
            log_like -= gammaln(np.sum(self.B_kw[k] + self.beta))
        self.loglike[iteration] = log_like
        # Copied over from the colgibbs_logpred.m MATLAB file.
        P_dk = self.alpha + self.A_dk
        ss = P_dk.sum(axis=1, keepdims=True)
        P_dk = P_dk / ss
        P_kw = self.beta + self.B_kw
        ss = P_kw.sum(axis=1, keepdims=True)
        P_kw = P_kw / ss
        log_like = self.docs_words_test.ravel().dot(np.log(P_dk@P_kw).ravel())
        self.loglike_test[iteration] = log_like

    def to_return_from_run(self):
        return self.doc_word_samples


def read_data(filename):
    """
    Reads the text data and splits into train/test.
    Examples:
    docs_words_train, docs_words_test = read_data('./code/toyexample.data')
    nips_train, nips_test = read_data('./code/nips.data')
    :param filename:    path to the file
    :return:
    docs_words_train:   training data, [n_docs, n_words] numpy array
    docs_words_test:    test data, [n_docs, n_words] numpy array
    """
    data = pd.read_csv(filename, dtype=int, sep=' ', names=['doc', 'word', 'train', 'test'])

    n_docs = np.amax(data.loc[:, 'doc'])
    n_words = np.amax(data.loc[:, 'word'])

    docs_words_train = np.zeros((n_docs, n_words), dtype=int)
    docs_words_test = np.zeros((n_docs, n_words), dtype=int)

    docs_words_train[data.loc[:, 'doc'] - 1, data.loc[:, 'word'] - 1] = data.loc[:, 'train']
    docs_words_test[data.loc[:, 'doc'] - 1, data.loc[:, 'word'] - 1] = data.loc[:, 'test']

    return docs_words_train, docs_words_test


def main():
    print('Running toyexample.data with the standard sampler')

    docs_words_train, docs_words_test = read_data('./toyexample.data')
    n_docs, n_words = docs_words_train.shape
    n_topics = 3
    alpha = 1
    beta = 1
    random_seed = 0

    sampler = GibbsSampler(
        n_docs=n_docs, n_topics=n_topics, n_words=n_words,
        alpha=alpha, beta=beta, random_seed=random_seed
    )

    topic_doc_words_distr, theta, phi = sampler.run(
        docs_words_train, docs_words_test,
        n_iter=200, save_loglike=True
    )

    print(phi * [phi > 1e-2])

    like_train, like_test = sampler.get_loglike()

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(like_train, color='blue')
    axs[0].set_ylabel('Log-joint', size=22)
    axs[0].set_title('Train Subset of Toy Data', size=22)
    axs[1].plot(like_test, color='green')
    axs[1].set_ylabel('Log-predictive', size=22)
    axs[1].set_xlabel('Iteration', size=22)
    axs[1].set_title('Test Subset of Toy Data', size=22)
    fig.tight_layout()
    plt.savefig('assets/lda/std-gibbs-toy.png')
    plt.close(fig)

    print('Running toyexample.data with the collapsed sampler')

    sampler_collapsed = GibbsSamplerCollapsed(
        n_docs=n_docs, n_topics=n_topics, n_words=n_words,
        alpha=alpha, beta=beta, random_seed=random_seed
    )

    doc_word_samples = sampler_collapsed.run(
        docs_words_train, docs_words_test,
        n_iter=200, save_loglike=True
    )
    
    topic_counts = np.zeros((3, 6))
    for doc in range(doc_word_samples.shape[0]):
        for word in range(doc_word_samples.shape[1]):
            for topic in doc_word_samples[doc, word]:
                topic_counts[topic, word] += 1

    print(topic_counts)

    like_train, like_test = sampler_collapsed.get_loglike()

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(like_train, color='blue')
    axs[0].set_ylabel('Log-joint', size=22)
    axs[0].set_title('Train Subset of Toy Data', size=22)
    axs[1].plot(like_test, color='green')
    axs[1].set_ylabel('Log-predictive', size=22)
    axs[1].set_xlabel('Iteration', size=22)
    axs[1].set_title('Test Subset of Toy Data', size=22)
    fig.tight_layout()
    plt.savefig('assets/lda/col-gibbs-toy.png')
    plt.close(fig)


if __name__ == '__main__':
    main()