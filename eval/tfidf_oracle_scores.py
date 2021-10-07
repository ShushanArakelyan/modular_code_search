import sys

import numpy as np
import pandas as pd
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import euclidean_distances

data = pd.read_json(
    '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_0.jsonl.gz',
    lines=True)
neg_data = pd.read_json(
    '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_5.jsonl.gz',
    lines=True)

# range_start = sys.argv[1]
# range_end = sys.argv[2]

range_start = 0
range_end = 30000

print("Range start: ", range_start, ", range end: ", range_end)

neg_count = 999
oracle_neg_count = 9
oracle_scores = []

for idx in tqdm.tqdm(range(int(range_start), int(range_end))):
    docs = [' '.join(data['code_tokens'][idx])]
    query = [' '.join(data['docstring_tokens'][idx])]
    np.random.seed(idx)
    negative_sample_idxs = [0]
    for _ in range(neg_count):
        random_idx = np.random.randint(1, len(neg_data), 1)[0]
        negative_sample_idxs.append(random_idx)
        docs.append(' '.join(neg_data['code_tokens'][random_idx]))

    tfIdfVectorizer = TfidfVectorizer(use_idf=True)
    tfIdfVectorizer.fit_transform(docs)
    tfIdf_docs = tfIdfVectorizer.transform(docs)
    tfIdf_query = tfIdfVectorizer.transform(query)

    distances = euclidean_distances(tfIdf_query, tfIdf_docs)
    top_idxs = np.argsort(distances[0, :])[:oracle_neg_count + 1]
    oracle_negative_samples = [negative_sample_idxs[i] for i in top_idxs]
    oracle_scores.append(oracle_negative_samples)
    
with open(f'/home/shushan/tfidf_oracle_scores/oracle_scores_{range_start}_{range_end}.txt', 'w') as f:
    for scores in oracle_scores:
        for score in scores:
            f.write(str(score))
            f.write(' ')
        f.write('\n')
