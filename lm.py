#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np
import pandas as pd
import collections
from math import log

from collections import defaultdict
from pdb import set_trace

from numpy.core.numeric import roll

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    SOS = 'START_OF_SENTENCE'
    UNK = 'UNK'
    EOS = 'END_OF_SENTENCE'
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        vocab_set = set(self.vocab())
        words_set  = set([w for s in corpus for w in s])
        numOOV = len(words_set - vocab_set)
        return pow(2.0, self.entropy(corpus, numOOV))

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s, numOOV)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence, numOOV):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i], numOOV)
        p += self.cond_logprob(LangModel.EOS, sentence, numOOV)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word(LangModel.EOS)

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
            return self.model[word]
        else:
            return self.lunk_prob - log(numOOV, 2)

    def vocab(self):
        return self.model.keys()

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_vocab(corpus, min_freq=2):
    vocab = set()
    unigrams = defaultdict(lambda: 0)
    for s in corpus:
        for word in s + [LangModel.EOS]:
            unigrams[word] += 1
            vocab.add(word)

    # Iterate through tokens seen in corpus
    for t_word in list(unigrams):
        if unigrams[t_word] < min_freq:
            vocab.remove(t_word)
    return vocab

def get_counts(n_max, corpus, vocab):
    ngram_counts = defaultdict(lambda: 0)
    start_ngram_counts = defaultdict(lambda: 0)
    for s in corpus:
        s = np.array([w if w in vocab else LangModel.UNK for w in s] + [LangModel.EOS])
        _n_max = min(len(s), n_max)
        for n in range(_n_max):
            for i, w in enumerate(rolling_window(s, n + 1)):
                t = tuple(w)
                if i == 0:
                    start_ngram_counts[t] += 1
                ngram_counts[t] += 1
    return ngram_counts, start_ngram_counts

def get_prob_numer_denom(n, t, sent_count, word_count, ngram_counts, start_ngram_counts):
    numer = 0
    denom = 0
    try:
        if n > len(t):
            if t in start_ngram_counts:
                numer += start_ngram_counts[t]
            if len(t) == 1:
                denom += sent_count
            elif t[:-1] in start_ngram_counts:
                denom += start_ngram_counts[t[:-1]]
        else:
            t = t[-n:]
            if t in ngram_counts:
                numer += ngram_counts[t]
            if len(t) == 1:
                denom += word_count
            elif t[:-1] in ngram_counts:
                denom += ngram_counts[t[:-1]]
    except Exception:
        set_trace()
    return numer, denom


def grid_search(lm_class, datasets, hyper_grid):
    from sklearn.model_selection import ParameterGrid
    grid = list(ParameterGrid(hyper_grid))
    results = pd.DataFrame(columns=list(hyper_grid.keys())
                           + [f'perplexity{j}' for j in range(len(datasets))])
    for i, params in enumerate(grid):
        print(params)
        lm = lm_class(**params)
        res_row = params
        for j, data in enumerate(datasets):
            lm.fit_corpus(data.train)
            if hasattr(lm, 'fit_dev'):
                lm.fit_dev(data.dev)
            res_row[f'perplexity{j}'] = lm.perplexity(data.dev)
        results.loc[i] = pd.Series(res_row)
    return results

class NGramLM(LangModel):
    def __init__(self, n=2, min_freq=2, lbd=1):
        # assert min_freq > 1
        # assert n == 1 or lbd > 0
        self.n = n
        self.min_freq = min_freq
        self.lbd = lbd if n > 1 else 0

    def grid_search(datasets):
        hyper_grid = {
            'n': [1, 2, 3],
            'min_freq': [2, 3, 4],
            'lbd': [0.0001, 0.001, 0.01, 0.1, 1]
        }
        results = []
        n_max = max(hyper_grid['n'])
        lm = NGramLM(n=n_max)
        param_idx_map = {}
        for lm.min_freq in hyper_grid['min_freq']:
            for dname, data in datasets.items():
                lm.fit_corpus(data.train)
                for lm.n in hyper_grid['n']:
                    lbds = [0] if lm.n == 1 else hyper_grid['lbd']
                    for lm.lbd in lbds:
                        perplexity = lm.perplexity(data.dev)
                        params = (lm.n, lm.min_freq, lm.lbd)
                        if params not in param_idx_map:
                            res_row = {
                                'n': lm.n,
                                'min_freq': lm.min_freq,
                                'lbd': lm.lbd,
                                dname: perplexity
                            }
                            param_idx_map[params] = len(results)
                            results.append(res_row)
                        else:
                            res_row = results[param_idx_map[params]]
                            res_row[dname] = perplexity
                        print(res_row)
        results_df = pd.DataFrame(results)
        results_df.sort_values(by=['n', 'min_freq', 'lbd'], ignore_index=True)
        return results_df

    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        self.sent_count = len(corpus)
        self.word_count = sum([len(s) + 1 for s in corpus])
        self._vocab = get_vocab(corpus=corpus, min_freq=self.min_freq)
        self.ngram_counts, self.start_ngram_counts = get_counts(n_max=self.n, corpus=corpus, vocab=self._vocab)

    def norm(self):
        pass

    def cond_logprob(self, word, previous, numOOV):
        n = self.n
        num_v = len(self._vocab) + 1        # +1 for UNK
        t = tuple(w if w in self._vocab else LangModel.UNK for w in previous + [word])
        numer, denom = get_prob_numer_denom(n=n, t=t, sent_count=self.sent_count,
                                            word_count=self.word_count,
                                            ngram_counts=self.ngram_counts,
                                            start_ngram_counts=self.start_ngram_counts)
        numer += self.lbd
        denom += self.lbd * num_v
        log_prob = np.log2(numer) - np.log2(denom)
        if t[-1] == LangModel.UNK:
            log_prob -= np.log2(numOOV)
        return log_prob

    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return self._vocab

class BackoffNGramLM(LangModel):
    def __init__(self, n_max=2, min_freq=2):
        self.n_max = n_max
        self.min_freq = min_freq

    def find_best_hyper(datasets):
        hyper_grid = {
            'n_max': [1, 2, 3],
            'min_freq': [2, 3, 4]
        }
        results = grid_search(NGramLM, datasets=datasets, hyper_grid=hyper_grid)
        return results

    def grid_search(datasets):
        hyper_grid = {
            'n': [2, 3, 4],
            'min_freq': [2, 3, 4, 5, 6]
        }
        results = []
        n_max = max(hyper_grid['n'])
        lm = BackoffNGramLM(n_max=n_max)
        param_idx_map = {}
        for lm.min_freq in hyper_grid['min_freq']:
            for dname, data in datasets.items():
                lm.fit_corpus(data.train)
                for lm.n in hyper_grid['n']:
                    perplexity = lm.perplexity(data.dev)
                    params = (lm.n, lm.min_freq)
                    if params not in param_idx_map:
                        res_row = {
                            'n': lm.n,
                            'min_freq': lm.min_freq,
                            dname: perplexity
                        }
                        param_idx_map[params] = len(results)
                        results.append(res_row)
                    else:
                        res_row = results[param_idx_map[params]]
                        res_row[dname] = perplexity
                    print(res_row)
        results_df = pd.DataFrame(results)
        results_df.sort_values(by=['n', 'min_freq'], ignore_index=True)
        return results_df

    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        self.sent_count = len(corpus)
        self.word_count = sum([len(s) + 1 for s in corpus])
        self._vocab = get_vocab(corpus=corpus, min_freq=self.min_freq)
        self.ngram_counts, self.start_ngram_counts = get_counts(n_max=self.n_max, corpus=corpus, vocab=self._vocab)

    def norm(self):
        pass

    def cond_logprob(self, word, previous, numOOV):
        t = tuple(w if w in self._vocab else LangModel.UNK for w in previous + [word])
        try:
            for n in range(self.n_max, 0, -1):
                numer, denom = get_prob_numer_denom(n=n, t=t, sent_count=self.sent_count,
                                                    word_count=self.word_count,
                                                    ngram_counts=self.ngram_counts,
                                                    start_ngram_counts=self.start_ngram_counts)
                if numer != 0:
                    assert(denom != 0)
                    log_prob = np.log2(numer) - np.log2(denom)
                    if t[-1] == LangModel.UNK:
                        log_prob -= np.log2(numOOV)
                    return log_prob
        except Exception:
            set_trace()

    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return self._vocab

class InterpolatedNGramLM(LangModel):
    def __init__(self, n_max=2, min_freq=2, lbd=None):
        self.n_max = n_max
        self.min_freq = min_freq
        self.lbd = lbd if lbd is not None else {n: 0.001 for n in range(1, n_max + 1)}

    def grid_search(datasets):
        hyper_grid = {
            'n': [2, 3],
            'min_freq': [2, 3, 4]
        }
        results = []
        n_max = max(hyper_grid['n'])
        lm = InterpolatedNGramLM(n_max=n_max)
        param_idx_map = {}
        for lm.min_freq in hyper_grid['min_freq']:
            for dname, data in datasets.items():
                lm.fit_corpus(data.train)
                for lm.n in hyper_grid['n']:
                    perplexity = lm.perplexity(data.dev)
                    params = (lm.n, lm.min_freq)
                    if params not in param_idx_map:
                        res_row = {
                            'n': lm.n,
                            'min_freq': lm.min_freq,
                            dname: perplexity
                        }
                        param_idx_map[params] = len(results)
                        results.append(res_row)
                    else:
                        res_row = results[param_idx_map[params]]
                        res_row[dname] = perplexity
                    print(res_row)
        results_df = pd.DataFrame(results)
        results_df.sort_values(by=['n', 'min_freq'], ignore_index=True)
        return results_df


    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        self.sent_count = len(corpus)
        self.word_count = sum([len(s) + 1 for s in corpus])
        self._vocab = get_vocab(corpus=corpus, min_freq=self.min_freq)
        self.ngram_counts, self.start_ngram_counts = get_counts(n_max=self.n_max, corpus=corpus, vocab=self._vocab)

    def norm(self):
        pass

    def cond_prob(self, n, word, previous, numOOV):
        try:
            num_v = len(self._vocab) + 1        # +1 for UNK
            t = tuple(w if w in self._vocab else LangModel.UNK for w in previous + [word])
            numer, denom = get_prob_numer_denom(n=n, t=t, sent_count=self.sent_count,
                            word_count=self.word_count,
                            ngram_counts=self.ngram_counts,
                            start_ngram_counts=self.start_ngram_counts)
            numer += self.lbd[n]
            denom += self.lbd[n] * num_v
            prob = numer / denom
            if t[-1] == LangModel.UNK:
                prob /= numOOV
        except:
            set_trace()
        return prob


    def get_dev_probs(self, dev_corpus):
        words_set  = set([w for s in dev_corpus for w in s])
        numOOV = len(words_set - self._vocab)
        dev_word_count = sum([len(s) + 1 for s in dev_corpus])
        probs = np.empty([dev_word_count, self.n_max])
        cnt = 0
        for s in dev_corpus:
            for i in range(len(s)):
                for n in range(1, self.n_max + 1):
                    probs[cnt, n - 1] = self.cond_prob(n, s[i], s[:i], numOOV)
                cnt += 1
            for n in range(1, self.n_max + 1):
                probs[cnt, n - 1] = self.cond_prob(n, LangModel.EOS, s, numOOV)
            cnt += 1
        assert(cnt == dev_word_count)
        return probs

    def fit_dev(self, dev_corpus, init_alpha=None, n_iter=10):
        dev_probs = self.get_dev_probs(dev_corpus)
        if init_alpha is not None:
            alpha = np.array(init_alpha)
        else:
            alpha = np.ones(self.n_max) / self.n_max
        print(alpha)
        for _ in range(n_iter):
            # E-step: calculate posterior probabilities from current model weights
            weighted_probs = dev_probs * alpha
            interpolated_probs = weighted_probs.sum(axis=1, keepdims=True)
            posterior_probs = weighted_probs / interpolated_probs

            # M-step: update model weights using posterior probabilities from E-step
            alpha = posterior_probs.mean(axis=0)
            print(alpha)
        self.alpha = alpha

    def cond_logprob(self, word, previous, numOOV):
        cond_probs = np.array([self.cond_prob(n, word, previous, numOOV)
                               for n in range(1, self.n_max + 1)])
        cond_prob = cond_probs.dot(self.alpha)
        return np.log2(cond_prob)

    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return self._vocab



# Single NGram
    # 1 - min_freq,
    # 2 - min_freq, lambda
    # 3 - min_freq, lambda
# Range of NGram
    # backoff
        # (1, 2) - min_freq
        # (1, 3) - min_freq
    # interpolation
        # (1, 2) - alpha, lambda
        # (1, 3) - alpha, lambda