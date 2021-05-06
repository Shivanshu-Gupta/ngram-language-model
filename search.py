from pdb import set_trace
import numpy as np
from data import read_texts
from lm import InterpolatedNGramLM, NGramLM, BackoffNGramLM

dnames = ["brown", "reuters", "gutenberg"]
datas = {}
results = []
# Learn the models for each of the domains, and evaluate it
for dname in dnames:
    print("-----------------------")
    print(dname)
    data = read_texts("data", dname)
    datas[dname] = data

results_ngramlm = NGramLM.grid_search(datas)
results_backoff = BackoffNGramLM.grid_search(datas)
results_interpolation = InterpolatedNGramLM

