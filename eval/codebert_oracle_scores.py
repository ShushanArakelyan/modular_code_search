import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import os
import sys
import torch
import tqdm
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification)
from transformers import AutoTokenizer
import codebert_embedder as embedder
from eval.utils import mrr

device = 'cuda:0'
model_dir = '/home/anna/CodeBERT/CodeBERT/codesearch/models/'
model = RobertaForSequenceClassification.from_pretrained(model_dir + "python_reduced_codebert")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = model.to(device)

data = pd.read_json('/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_0.jsonl.gz', lines=True)
neg_data = pd.read_json('/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_5.jsonl.gz', lines=True)
embedder.init_embedder('cuda:0')

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
    for _ in range(neg_count):
        random_idx = np.random.randint(0, len(neg_data), 1)[0]
        sample = (data['docstring_tokens'][idx],
                  neg_data['code_tokens'][random_idx],
                  neg_data['static_tags'][random_idx],
                  neg_data['regex_tags'][random_idx],
                  data['ccg_parse'][idx])
        all_samples.append(sample)
        docs.append(' '.join(neg_data['code_tokens'][random_idx]))

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
    oracle_scores.append(np.argsort(preds[:, 1])[::-1][:10])
with open(f'/home/shushan/oracle_scores_{range_start}_{range_end}.txt', 'w') as f:
    for scores in oracle_scores:
        for score in scores:
            f.write(str(score))
            f.write(' ')
        f.write('\n')
