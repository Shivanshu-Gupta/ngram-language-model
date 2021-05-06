from pdb import set_trace
import numpy as np
from data import read_texts, learn_unigram, print_table

import warnings
warnings.filterwarnings("error")

def learn_ngramlm(data):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import NGramLM, BackoffNGramLM, InterpolatedNGramLM
    # ngramlm = NGramLM(n=3, lbd=0.001, min_freq=5)
    # ngramlm = BackoffNGramLM(n_max=4, min_freq=5)
    ngramlm = InterpolatedNGramLM(n_max=2, min_freq=4, lbd={1: 0, 2: 0.01})
    # ngramlm = InterpolatedNGramLM(n_max=3, min_freq=4, lbd={1: 0, 2: 0.01, 3: 0.001})

    # set_trace()
    ngramlm.fit_corpus(data.train)
    if hasattr(ngramlm, 'fit_dev'):
        ngramlm.fit_dev(data.dev)
    print("vocab:", len(ngramlm.vocab()))
    # evaluate on train, test, and dev
    print("train:", ngramlm.perplexity(data.train))
    # set_trace()
    print("dev  :", ngramlm.perplexity(data.dev))
    print("test :", ngramlm.perplexity(data.test))
    from generator import Sampler
    sampler = Sampler(ngramlm)
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    return ngramlm

dnames = ["brown", "reuters", "gutenberg"]
datas = []
models = []
# Learn the models for each of the domains, and evaluate it
for dname in dnames:
    print("-----------------------")
    print(dname)
    data = read_texts("data", dname)
    datas.append(data)
    # model = learn_unigram(data)
    model = learn_ngramlm(data)
    models.append(model)
# compute the perplexity of all pairs
n = len(dnames)
perp_dev = np.zeros((n,n))
perp_test = np.zeros((n,n))
perp_train = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        perp_dev[i][j] = models[i].perplexity(datas[j].dev)
        perp_test[i][j] = models[i].perplexity(datas[j].test)
        perp_train[i][j] = models[i].perplexity(datas[j].train)

print("-------------------------------")
print("x train")
print_table(perp_train, dnames, dnames, "results/table-train.tex")
print("-------------------------------")
print("x dev")
print_table(perp_dev, dnames, dnames, "results/table-dev.tex")
print("-------------------------------")
print("x test")
print_table(perp_test, dnames, dnames, "results/table-test.tex")

