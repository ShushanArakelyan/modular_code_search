import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class CodeSearchNetDataset(Dataset):
    def __init__(self, data_dir, file_it, device):
        self.data = pd.read_json(f'{data_dir}/ccg_train_{file_it}.jsonl.gz', lines=True)
        data_dir = '/home/shushan/train' # TODO - move all preprocessed scoring files to the datasets directory;
        self.device = device
        self.scores_data_memmap = np.memmap(f'{data_dir}/memmap_scores_data_{file_it}.npy', dtype='float32', mode='r', shape=(200000, 512, 1))
        self.scores_offsets_memmap = np.memmap(f'{data_dir}/memmap_scores_offsets_{file_it}.npy', dtype='int32', mode='r', shape=(60000, 1))
        self.verbs_data_memmap = np.memmap(f'{data_dir}/memmap_verbs_data_{file_it}.npy', dtype='float32', mode='r', shape=(200000, 1, 768))
        self.verbs_offsets_memmap = np.memmap(f'{data_dir}/memmap_verbs_offsets_{file_it}.npy', dtype='int32', mode='r', shape=(60000, 1))
        self.code_data_memmap = np.memmap(f'{data_dir}/memmap_code_data_{file_it}.npy', dtype='float32', mode='r', shape=(200000, 512, 768))
        self.code_offsets_memmap = np.memmap(f'{data_dir}/memmap_code_offsets_{file_it}.npy', dtype='int32', mode='r', shape=(60000, 1))
        self.positive_label = torch.FloatTensor([1]).to(device)
        self.negative_label = torch.FloatTensor([0]).to(device)
    
    def __len__(self):
        return 2 * len(self.data) - 1 # TODO: temporary -1 until redo precomputed scores

    def __getitem__(self, idx):
        if idx < len(self.data):
            sample = (self.data['docstring_tokens'][idx],
                      self.data['code_tokens'][idx],
                      self.data['static_tags'][idx],
                      self.data['regex_tags'][idx],
                      self.data['ccg_parse'][idx])
            label = self.positive_label
        else: # is a negative samples
            i = idx - len(self.data)
            np.random.seed(i)
            random_idx = np.random.randint(0, len(self.data), 1)[0]
            sample = (self.data['docstring_tokens'][i],
                      self.data['code_tokens'][random_idx],
                      self.data['static_tags'][random_idx],
                      self.data['regex_tags'][random_idx],
                      self.data['ccg_parse'][i])
            label = self.negative_label
        scores_start = self.scores_offsets_memmap[idx][0]
        scores_end = self.scores_offsets_memmap[idx + 1][0]
        scores = torch.FloatTensor(self.scores_data_memmap[scores_start:scores_end]).to(self.device)
        verbs_start = self.verbs_offsets_memmap[idx][0]
        verbs_end = self.verbs_offsets_memmap[idx + 1][0]
        verbs = torch.FloatTensor(self.verbs_data_memmap[verbs_start:verbs_end]).to(self.device)
        code_start = self.code_offsets_memmap[idx][0]
        code_end = self.code_offsets_memmap[idx + 1][0]
        code_embeddings = torch.FloatTensor(self.code_data_memmap[code_start:code_end]).to(self.device)

        return (sample, scores, verbs, code_embeddings, label)
