import sys

import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer
from transformers import (RobertaForSequenceClassification)

import codebert_embedder as embedder

device = 'cuda:0'
embedder.init_embedder(device)
model_dir = '/home/anna/CodeBERT/CodeBERT/codesearch/models/'
embedder.model = RobertaForSequenceClassification.from_pretrained(model_dir + "python_cosqa_codebert")
embedder.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
embedder.model = embedder.model.to(device)

data = pd.read_json(
    '/home/shushan/datasets/CoSQA/resources/ccg_parses_only/python/final/jsonl/train/ccg_cosqa_parsed.jsonl.gz',
    lines=True)

range_start = sys.argv[1]
range_end = sys.argv[2]
range_end = min(len(data), int(range_end))

print("Range start: ", range_start, ", range end: ", range_end)

with open(f'/home/shushan/codebert_oracle_scores_cosqa/oracle_scores_{range_start}_{range_end}.txt', 'w') as f:
    embedder.model.eval()
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
        np.random.seed(idx)
        negative_sample_idxs = []
        for _ in range(neg_count):
            random_idx = idx
            while random_idx == idx:
                random_idx = np.random.randint(1, len(data), 1)[0]
            negative_sample_idxs.append(random_idx)
            sample = (data['docstring_tokens'][idx],
                      data['code_tokens'][random_idx],
                      data['static_tags'][random_idx],
                      data['regex_tags'][random_idx],
                      data['ccg_parse'][idx])
            all_samples.append(sample)

    #     print("negative sample idxs: ", negative_sample_idxs)
        preds = None
        with torch.no_grad():
            for s in all_samples:
                query, code = s[0], s[1]
                inputs = embedder.get_feature_inputs(' '.join(query), ' '.join(code))
                outputs = embedder.model(**inputs)
                logits = outputs[0]
                if preds is None:
                    preds = logits.cpu().numpy()
                else:
                    preds = np.append(preds, logits.cpu().numpy(), axis=0)
        top_idxs = np.argsort(preds[:, 1])[::-1][:oracle_neg_count + 1]
    #     print("top idxs: ", top_idxs)
    #     print("hardest examples: ", [negative_sample_idxs[idx - 1] if idx >= 1 else 0 for idx in top_idxs])
        top_examples = [negative_sample_idxs[idx - 1] if idx >= 1 else 0 for idx in top_idxs]
        oracle_scores.append(top_examples)
        for eg in top_examples:
            f.write(str(eg))
            f.write(' ')
        f.write('\n')
        f.flush()



