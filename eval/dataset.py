import os
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import euclidean_distances
from torch.utils.data import Dataset
import numpy as np


def transform_sample(sample):
    nsample = []
    for si in sample[:-1]:
        nsii = []
        for sii in si:
            if len(sii) > 0:
                nsii.append(sii[0])
            else:
                nsii.append(sii)
        nsample.append(nsii)
    ccg_parse = sample[-1][0][1:-1]
    return ccg_parse, nsample

    
class CodeSearchNetDataset(Dataset):
    def __init__(self, data_dir, file_it, device):
        self.device = device
        
        print('loading ', f'{data_dir}/ccg_train_{file_it}.jsonl.gz')
        self.data = pd.read_json(f'{data_dir}/ccg_train_{file_it}.jsonl.gz', lines=True)
        offsets_count = int(len(self.data) + 1)
        data_map_count = int(len(self.data) * 3.5)

        self.scores_data_memmap = np.memmap(f'{data_dir}/memmap_scores_data_{file_it}.npy', dtype='float32', mode='r', shape=(data_map_count, 512, 1))
        self.scores_offsets_memmap = np.memmap(f'{data_dir}/memmap_scores_offsets_{file_it}.npy', dtype='int32', mode='r', shape=(offsets_count, 1))
        self.verbs_data_memmap = np.memmap(f'{data_dir}/memmap_verbs_data_{file_it}.npy', dtype='float32', mode='r', shape=(data_map_count, 1, 768))
        self.verbs_offsets_memmap = np.memmap(f'{data_dir}/memmap_verbs_offsets_{file_it}.npy', dtype='int32', mode='r', shape=(offsets_count, 1))
        self.code_data_memmap = np.memmap(f'{data_dir}/memmap_code_data_{file_it}.npy', dtype='float32', mode='r', shape=(data_map_count, 512, 768))
        self.code_offsets_memmap = np.memmap(f'{data_dir}/memmap_code_offsets_{file_it}.npy', dtype='int32', mode='r', shape=(offsets_count, 1))
        self.positive_label = torch.FloatTensor([1]).to(device)
        self.negative_label = torch.FloatTensor([0]).to(device)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.data['docstring_tokens'][idx],
                  self.data['code_tokens'][idx],
                  self.data['static_tags'][idx],
                  self.data['regex_tags'][idx],
                  self.data['ccg_parse'][idx])
        label =  self.positive_label if (self.data['label'][idx] == 1) else self.negative_label
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


class CodeSearchNetDataset_BalancedNegatives(Dataset):
    def __init__(self, data_dir, file_it, device, num_negatives=1):
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
    

class CodeSearchNetDataset_NotPrecomputed(Dataset):
    def __init__(self, filename, device, neg_count):
        self.data = pd.read_json(filename, lines=True)
        self.device = device
        self.neg_count = neg_count
        self.positive_label = torch.FloatTensor([1]).to(device)
        self.negative_label = torch.FloatTensor([0]).to(device)

    def __len__(self):
        return len(self.data)  # TODO: temporary -1 until redo precomputed scores

    def __getitem__(self, idx):
        all_samples = []
        sample = (self.data['docstring_tokens'][idx],
                  self.data['code_tokens'][idx],
                  self.data['static_tags'][idx],
                  self.data['regex_tags'][idx],
                  self.data['ccg_parse'][idx])
        np.random.seed(idx)
        all_samples.append(sample)
        for _ in range(self.neg_count):
            random_idx = np.random.randint(0, len(self.data), 1)[0]
            # if we sampled the correct idx, sample again
            while random_idx == idx:
                random_idx = np.random.randint(0, len(self.data), 1)[0]
            sample = (self.data['docstring_tokens'][idx],
                      self.data['code_tokens'][random_idx],
                      self.data['static_tags'][random_idx],
                      self.data['regex_tags'][random_idx],
                      self.data['ccg_parse'][idx])
            all_samples.append(sample)
        return all_samples


class CodeSearchNetDataset_TFIDFOracle(Dataset):
    def __init__(self, filename, device, neg_count, oracle_neg_count):
        self.data = pd.read_json(filename, lines=True)
        self.device = device
        self.neg_count = neg_count
        self.oracle_neg_count = oracle_neg_count
        self.positive_label = torch.FloatTensor([1]).to(device)
        self.negative_label = torch.FloatTensor([0]).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_samples = []
        sample = (self.data['docstring_tokens'][idx],
                  self.data['code_tokens'][idx],
                  self.data['static_tags'][idx],
                  self.data['regex_tags'][idx],
                  self.data['ccg_parse'][idx])
        all_samples.append(sample)
        docs = [' '.join(self.data['code_tokens'][idx])]
        query = [' '.join(self.data['docstring_tokens'][idx])]
        np.random.seed(idx)
        for _ in range(self.neg_count):
            random_idx = np.random.randint(0, len(self.data), 1)[0]
            # if we sampled the correct idx, sample again
            while random_idx == idx:
                random_idx = np.random.randint(0, len(self.data), 1)[0]
            sample = (self.data['docstring_tokens'][idx],
                      self.data['code_tokens'][random_idx],
                      self.data['static_tags'][random_idx],
                      self.data['regex_tags'][random_idx],
                      self.data['ccg_parse'][idx])
            all_samples.append(sample)
            docs.append(' '.join(self.data['code_tokens'][random_idx]))

        tfIdfVectorizer = TfidfVectorizer(use_idf=True)
        tfIdfVectorizer.fit_transform(docs)
        tfIdf_docs = tfIdfVectorizer.transform(docs)
        tfIdf_query = tfIdfVectorizer.transform(query)

        distances = euclidean_distances(tfIdf_query, tfIdf_docs)
        correct_distance = distances[:, 0]
        rank = np.sum(distances[:, 1:] < correct_distance)
        if rank >= self.oracle_neg_count + 1:
            oracle_idxs = [0]
            oracle_idxs.extend(np.argsort(distances[:, 1:])[0, :self.oracle_neg_count])
        else:
            # sort indices so the correct one is in index 0
            oracle_idxs = sorted(np.argsort(distances[:, 1:])[0, :self.oracle_neg_count + 1])
        oracle_negative_samples = [all_samples[i] for i in oracle_idxs]
        return oracle_negative_samples