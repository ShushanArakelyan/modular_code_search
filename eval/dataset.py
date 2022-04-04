import glob
import natsort
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from layout_assembly.precompute_scoring_in_shards import CodeSearchNetDataset_Random_NegOnly


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


def filter_neg_samples(dataset, device):
    class FilteredDataset(CodeSearchNetDataset):
        def __init__(self, device, data):
            self.data = data
            self.positive_label = torch.FloatTensor([1]).to(device)
            self.negative_label = torch.FloatTensor([0]).to(device)

        def __getitem__(self, item):
            sample, scores, verbs, label = self.data[item]
            label = self.positive_label if (label == 1) else self.negative_label
            return sample, scores, verbs, label

    new_data = []
    for i in range(len(dataset)):
        sample, _, _, label = dataset[i]
        if label == 1:
            new_data.append((sample, None, None, label))
    return FilteredDataset(device, new_data)


class CodeSearchNetDataset(Dataset):
    def __init__(self, data_dir, file_it, device):
        self.device = device

        print('loading ', f'{data_dir}/ccg_train_{file_it}.jsonl.gz')
        self.data = pd.read_json(f'{data_dir}/ccg_train_{file_it}.jsonl.gz', lines=True)
        offsets_count = int(len(self.data) + 1)
        data_map_count = int(len(self.data) * 3.5)

        self.scores_data_memmap = np.memmap(f'{data_dir}/memmap_scores_data_{file_it}.npy', dtype='float32', mode='r',
                                            shape=(data_map_count, 512, 1))
        self.scores_offsets_memmap = np.memmap(f'{data_dir}/memmap_scores_offsets_{file_it}.npy', dtype='int32',
                                               mode='r', shape=(offsets_count, 1))
        self.verbs_data_memmap = np.memmap(f'{data_dir}/memmap_verbs_data_{file_it}.npy', dtype='float32', mode='r',
                                           shape=(data_map_count, 1, 768))
        self.verbs_offsets_memmap = np.memmap(f'{data_dir}/memmap_verbs_offsets_{file_it}.npy', dtype='int32', mode='r',
                                              shape=(offsets_count, 1))
        self.positive_label = torch.FloatTensor([1]).to(device)
        self.negative_label = torch.FloatTensor([0]).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx)
        sample = (self.data['docstring_tokens'][idx],
                  self.data['code_tokens'][idx],
                  self.data['static_tags'][idx],
                  self.data['regex_tags'][idx],
                  self.data['ccg_parse'][idx])
        label = self.positive_label if (self.data['label'][idx] == 1) else self.negative_label
        scores_start = self.scores_offsets_memmap[idx][0]
        scores_end = self.scores_offsets_memmap[idx + 1][0]
        scores = self.scores_data_memmap[scores_start:scores_end]
        scores = torch.FloatTensor(scores).to(self.device)
        verbs_start = self.verbs_offsets_memmap[idx][0]
        verbs_end = self.verbs_offsets_memmap[idx + 1][0]
        verbs = self.verbs_data_memmap[verbs_start:verbs_end]
        verbs = torch.FloatTensor(verbs).to(self.device)
        return (sample, scores, verbs, label) # verb embeddings are not used anymore


class CodeSearchNetDataset_wShards(CodeSearchNetDataset):
    def __init__(self, data_dir, file_it, shard_it, device):
        self.device = device

        print('loading ', f'{data_dir}/ccg_train_{file_it}_{shard_it}.jsonl.gz')
        self.data = pd.read_json(f'{data_dir}/ccg_train_{file_it}_{shard_it}.jsonl.gz', lines=True)
        offsets_count = int(len(self.data) + 1)
        data_map_count = int(len(self.data) * 3.5)

        self.scores_data_memmap = np.memmap(f'{data_dir}/memmap_scores_data_{file_it}_{shard_it}.npy', dtype='float32',
                                            mode='r', shape=(data_map_count, 512, 1))
        self.scores_offsets_memmap = np.memmap(f'{data_dir}/memmap_scores_offsets_{file_it}_{shard_it}.npy',
                                               dtype='int32', mode='r', shape=(offsets_count, 1))
        self.verbs_data_memmap = np.memmap(f'{data_dir}/memmap_verbs_data_{file_it}_{shard_it}.npy', dtype='float32',
                                           mode='r', shape=(data_map_count, 1, 768))
        self.verbs_offsets_memmap = np.memmap(f'{data_dir}/memmap_verbs_offsets_{file_it}_{shard_it}.npy',
                                              dtype='int32', mode='r', shape=(offsets_count, 1))
        self.positive_label = torch.FloatTensor([1]).to(device)
        self.negative_label = torch.FloatTensor([0]).to(device)


class CodeSearchNetDataset_NotPrecomputed(Dataset):
    def __init__(self, filename, device):
        self.data = pd.read_json(filename, lines=True)
        self.device = device
        self.positive_label = torch.FloatTensor([1]).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.data['docstring_tokens'][idx],
                  self.data['alt_code_tokens'][idx],
                  self.data['static_tags'][idx],
                  self.data['regex_tags'][idx],
                  self.data['ccg_parse'][idx])
        return sample, 1, 1, self.positive_label


class CodeSearchNetDataset_NotPrecomputed_RandomNeg(CodeSearchNetDataset_Random_NegOnly):
    def __init__(self, filename, device, range, length=30000):
        CodeSearchNetDataset_Random_NegOnly.__init__(self, filename=filename, device=device, neg_count=9, length=length)
        self.r = range
    def __getitem__(self, idx):
        all_samples = []
        all_samples.extend(self.get_random_idxs(idx, all_samples))
        sample = all_samples[self.r]
        return sample, 1, 1, self.negative_label

# class CodeSearchNetDataset_TFIDFOracle(Dataset):
#     def __init__(self, filename, device, neg_count, oracle_neg_count):
#         self.data = pd.read_json(filename, lines=True)
#         self.neg_data = pd.read_json(
#             '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_5.jsonl.gz',
#             lines=True)
#         self.device = device
#         self.neg_count = neg_count
#         self.oracle_neg_count = oracle_neg_count
#         self.positive_label = torch.FloatTensor([1]).to(device)
#         self.negative_label = torch.FloatTensor([0]).to(device)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         all_samples = []
#         sample = (self.data['docstring_tokens'][idx],
#                   self.data['alt_code_tokens'][idx],
#                   self.data['static_tags'][idx],
#                   self.data['regex_tags'][idx],
#                   self.data['ccg_parse'][idx])
#         all_samples.append(sample)
#         docs = [' '.join(self.data['code_tokens'][idx])]
#         query = [' '.join(self.data['docstring_tokens'][idx])]
#         np.random.seed(idx)
#         for _ in range(self.neg_count):
#             random_idx = np.random.randint(0, len(self.neg_data), 1)[0]
#             # if we sampled the correct idx, sample again
#             #             while random_idx == idx:
#             #                 random_idx = np.random.randint(0, len(self.neg_data), 1)[0]
#             sample = (self.data['docstring_tokens'][idx],
#                       self.neg_data['alt_code_tokens'][random_idx],
#                       self.neg_data['static_tags'][random_idx],
#                       self.neg_data['regex_tags'][random_idx],
#                       self.data['ccg_parse'][idx])
#             all_samples.append(sample)
#             docs.append(' '.join(self.neg_data['code_tokens'][random_idx]))
#
#         tfIdfVectorizer = TfidfVectorizer(use_idf=True)
#         tfIdfVectorizer.fit_transform(docs)
#         tfIdf_docs = tfIdfVectorizer.transform(docs)
#         tfIdf_query = tfIdfVectorizer.transform(query)
#
#         distances = euclidean_distances(tfIdf_query, tfIdf_docs)
#         correct_distance = distances[:, 0]
#         rank = np.sum(distances[:, 1:] < correct_distance)
#         if rank >= self.oracle_neg_count + 1:
#             oracle_idxs = [0]
#             oracle_idxs.extend(np.argsort(distances[:, 1:])[0, :self.oracle_neg_count])
#         else:
#             # sort indices so the correct one is in index 0
#             oracle_idxs = sorted(np.argsort(distances[:, 1:])[0, :self.oracle_neg_count + 1])
#         oracle_negative_samples = [all_samples[i] for i in oracle_idxs]
#         return oracle_negative_samples
#
#
class CodeSearchNetDataset_NegativeOracleNotPrecomputed(Dataset):
    def __init__(self, filename, device, neg_count, oracle_idxs_file):
        self.data = pd.read_json(filename, lines=True)
        self.device = device
        self.negative_count = neg_count
        self.negative_samples = {}
        self.read_oracle_idxs_from_file(oracle_idxs_file)
        self.negative_label = torch.FloatTensor([0]).to(device)

    def __len__(self):
        return len(self.negative_samples) * self.negative_count

    def __getitem__(self, idx):
        neg_idx = int(idx / len(self.negative_samples))
        idx = idx % len(self.negative_samples)
        print(idx, neg_idx)
        neg_idx = self.negative_samples[idx][neg_idx]
        sample = (self.data['docstring_tokens'][idx],
                  self.data['alt_code_tokens'][neg_idx],
                  self.data['static_tags'][neg_idx],
                  self.data['regex_tags'][neg_idx],
                  self.data['ccg_parse'][idx])
        return sample, 1, 1, self.negative_label

    def read_oracle_idxs_from_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f.readlines()):
                scores = line.strip('\n').split(' ')
                scores = [int(s) for s in scores if len(s) > 0 and int(s) > 0]
                self.negative_samples[i] = scores[:self.negative_count]