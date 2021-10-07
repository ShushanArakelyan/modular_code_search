import sys

import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer
from transformers import (RobertaForSequenceClassification)

import codebert_embedder as embedder

device = 'cuda:0'
model_dir = '/home/anna/CodeBERT/CodeBERT/codesearch/models/'
model = RobertaForSequenceClassification.from_pretrained(model_dir + "python_reduced_codebert")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = model.to(device)

data = pd.read_json(
    '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_0.jsonl.gz',
    lines=True)
neg_data = pd.read_json(
    '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_5.jsonl.gz',
    lines=True)
embedder.init_embedder(device)

range_start = sys.argv[1]
range_end = sys.argv[2]

print("Range start: ", range_start, ", range end: ", range_end)

model.eval()
neg_count = 999
oracle_neg_count = 9
oracle_scores = []
for idx in tqdm.tqdm(range(int(range_start), int(range_end))):
    all_samples = []
    sample = (data['docstring_tokens'][idx],
              data['code_tokens'][idx],
              data['static_tags'][idx],
              data['regex_tags'][idx],
              data['ccg_parse'][idx])
    all_samples.append(sample)
    docs = [' '.join(data['code_tokens'][idx])]
    query = [' '.join(data['docstring_tokens'][idx])]
    np.random.seed(idx)
    negative_sample_idxs = []
    for _ in range(neg_count):
        random_idx = np.random.randint(1, len(neg_data), 1)[0]
        negative_sample_idxs.append(random_idx)
        sample = (data['docstring_tokens'][idx],
                  neg_data['code_tokens'][random_idx],
                  neg_data['static_tags'][random_idx],
                  neg_data['regex_tags'][random_idx],
                  data['ccg_parse'][idx])
        all_samples.append(sample)
        docs.append(' '.join(neg_data['code_tokens'][random_idx]))
    
    print("negative sample idxs: ", negative_sample_idxs)
    preds = None
    with torch.no_grad():
        for s in all_samples:
            query, code = s[0], s[1]
            inputs = embedder.get_feature_inputs(' '.join(query), ' '.join(code))
            outputs = model(**inputs)
            logits = outputs[0]
            if preds is None:
                preds = logits.cpu().numpy()
            else:
                preds = np.append(preds, logits.cpu().numpy(), axis=0)
    top_idxs = np.argsort(preds[:, 1])[::-1][:10]
    print("hardest examples: ", [negative_sample_idxs[i] for i in top_idxs])
    oracle_scores.append([negative_sample_idxs[i] for i in top_idxs])
    
with open(f'/home/shushan/oracle_scores_{range_start}_{range_end}.txt', 'w') as f:
    for scores in oracle_scores:
        for score in scores:
            f.write(str(score))
            f.write(' ')
        f.write('\n')
