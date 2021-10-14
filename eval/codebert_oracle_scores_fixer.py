import sys
sys.path.append('/home/shushan/modular_code_search')

import glob
import natsort
import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer
from transformers import (RobertaForSequenceClassification)

import codebert_embedder as embedder

datadir = '/home/shushan/codebert_oracle_scores'
neg_data = pd.read_json('/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_5.jsonl.gz', lines=True)

neg_count = 999
oracle_idxs = [[] for _ in range(30000)]
all_score_files = glob.glob(datadir + '/*')
all_score_files = natsort.natsorted(all_score_files)
for score_file in all_score_files:
    print("Loading from file: ", score_file)
    parts = score_file.split('/')[-1].split('_')
    start = int(parts[-2])
    end = int(parts[-1].split('.')[0])
    with open(score_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            top_idxs = line.strip('\n').split(' ')
            top_idxs = [int(s) for s in top_idxs if len(s) > 0]
            np.random.seed(i)
            negative_sample_idxs = []
            for _ in range(neg_count):
                random_idx = np.random.randint(1, len(neg_data), 1)[0]
                negative_sample_idxs.append(random_idx) 
            oracle_idxs[start + i] = [negative_sample_idxs[idx - 1] if idx >= 1 else 0 for idx in top_idxs]
            
            
with open(f'/home/shushan/oracle_scores_0_30000.txt', 'w') as f:
    for scores in oracle_idxs:
        for score in scores:
            f.write(str(score))
            f.write(' ')
        f.write('\n')