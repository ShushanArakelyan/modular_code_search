import argparse
import glob
import os

import natsort
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset

import codebert_embedder as embedder
from layout_assembly.layout import LayoutNet
from layout_assembly.modules import ScoringModule, ActionModuleFacade


class CodeSearchNetDataset_SavedOracle_NegOnly(Dataset):
    def __init__(self, filename, device, neg_count, oracle_idxs):
        self.data = pd.read_json(filename, lines=True)
        self.neg_data = pd.read_json(
            '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_5.jsonl.gz',
            lines=True)
        self.device = device
        self.neg_count = neg_count
        self.oracle_idxs = []
        self.positive_label = torch.FloatTensor([1]).to(device)
        self.negative_label = torch.FloatTensor([0]).to(device)
        self.read_oracle_idxs(oracle_idxs)

    def __len__(self):
        return len(
            self.oracle_idxs)  # TODO: currently we sometimes only have the part of all indexes processed and saved

    def __getitem__(self, idx):
        all_samples = []
        all_samples.extend(self.get_oracle_idxs_or_random(idx, all_samples))
        return all_samples

    # TODO: remove random part when we have finished processing oracle
    # indexes for all of the training data
    def get_oracle_idxs_or_random(self, idx, samples):
        if len(self.oracle_idxs[idx]) == 0:
            oracle_idxs = []
            while len(oracle_idxs) < self.neg_count:
                random_idx = np.random.randint(0, len(self.neg_data), 1)[0]
                if random_idx != idx:
                    oracle_idxs.append(random_idx)
        else:
            oracle_idxs = np.asarray(self.oracle_idxs[idx])
            oracle_idxs = oracle_idxs[oracle_idxs != 0]
            # 9 is the max value for neg_count, but real neg_count can be different
            assert len(oracle_idxs) <= (9 + 1)
            if len(oracle_idxs) == 9 + 1:
                oracle_idxs = oracle_idxs[:-1]
        for i in range(self.neg_count):
            neg_idx = oracle_idxs[i]
            sample = (self.data['docstring_tokens'][idx],
                      self.neg_data['alt_code_tokens'][neg_idx],
                      self.neg_data['static_tags'][neg_idx],
                      self.neg_data['regex_tags'][neg_idx],
                      self.data['ccg_parse'][idx])
            samples.append(sample)
        return samples

    def read_oracle_idxs(self, oracle_idxs):
        if os.path.isdir(oracle_idxs):
            self.oracle_idxs = [[] for _ in range(30000)]
            all_score_files = glob.glob(oracle_idxs + '/*')
            all_score_files = natsort.natsorted(all_score_files)
            for score_file in all_score_files:
                print("Loading from file: ", score_file)
                parts = score_file.split('/')[-1].split('_')
                start = int(parts[-2])
                end = int(parts[-1].split('.')[0])
                self.read_oracle_idxs_from_file(start, score_file)
        else:
            self.oracle_idxs = [[] for _ in range(500)]  # TODO: fix me
            self.read_oracle_idxs_from_file(0, oracle_idxs)

    def read_oracle_idxs_from_file(self, start, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f.readlines()):
                scores = line.strip('\n').split(' ')
                scores = [int(s) for s in scores if len(s) > 0]
                self.oracle_idxs[start + i] = (scores)


device = 'cuda:0 '
scoring_checkpoint = "/home/shushan/finetuned_scoring_models/06-09-2021 20:21:51/model_3_ep_5.tar"


def sample_hard(idx, distances):
    neg_idx = np.argsort(distances[idx])[1]  # the zero-th element is idx itself
    distances[idx, neg_idx] = np.inf
    return neg_idx, distances


def sample_random(idx, data):
    random_idx = np.random.randint(0, len(data), 1)[0]
    return random_idx


def main(num_negatives, neg_sampling_strategy, shard_size):
    for file_it in range(1):
        data_dir = f'/home/shushan/train_v2_neg_{num_negatives}_{neg_sampling_strategy}'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_dir1 = '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train'
        data_file = f'{data_dir1}/ccg_train_{file_it}.jsonl.gz'
        if neg_sampling_strategy == 'codebert':
            dataset = CodeSearchNetDataset_SavedOracle_NegOnly(data_file, device,
                                                               oracle_idxs='/home/shushan/codebert_oracle_scores',
                                                               neg_count=num_negatives)
        elif neg_sampling_strategy == 'random':
            dataset = CodeSearchNetDataset_NotPrecomputed(data_file, device, neg_count=num_negatives)
        elif neg_sampling_strategy == 'hard':
            dataset = CodeSearchNetDataset_TFIDFOracle(data_file, device, neg_count=999, oracle_neg_count=num_negatives)

        scoring_module = ScoringModule(device, scoring_checkpoint)
        version = 1
        action_module = ActionModuleFacade(device, version, normalized=False)
        layout_net = LayoutNet(scoring_module, action_module, device)

        positive = torch.FloatTensor([[1]]).to(device)
        negative = torch.FloatTensor([[0]]).to(device)

        offsets_count = int(shard_size + 1)
        data_map_count = int(shard_size * 3.5)

        #         if neg_sampling_strategy == 'hard':
        #             tfIdfVectorizer = TfidfVectorizer(use_idf=True)
        #             tfIdfVectorizer.fit_transform([' '.join(dt) for dt in data['docstring_tokens']] + [' '.join(ct) for ct in data['code_tokens']])
        #             tfIdf_docs = tfIdfVectorizer.transform([' '.join(dt) for dt in data['docstring_tokens']])
        #             tfIdf_codes = tfIdfVectorizer.transform([' '.join(ct) for ct in data['code_tokens']])
        #             distances = euclidean_distances(tfIdf_docs, tfIdf_codes)
        for shard_it in range(1, num_negatives + 1):
            scores_data_map = np.memmap(f'{data_dir}/memmap_scores_data_{file_it}_{shard_it}.npy',
                                        dtype='float32', mode='w+', shape=(data_map_count, 512, 1))
            scores_offsets_map = np.memmap(f'{data_dir}/memmap_scores_offsets_{file_it}_{shard_it}.npy',
                                           dtype='int32', mode='w+', shape=(offsets_count, 1))

            verbs_data_map = np.memmap(f'{data_dir}/memmap_verbs_data_{file_it}_{shard_it}.npy',
                                       dtype='float32', mode='w+', shape=(data_map_count, 1, 768))
            verbs_offsets_map = np.memmap(f'{data_dir}/memmap_verbs_offsets_{file_it}_{shard_it}.npy',
                                          dtype='int32', mode='w+', shape=(offsets_count, 1))

            scores_offset = 0
            verbs_offset = 0
            new_scores_offset = 0
            new_verbs_offset = 0

            docstring_tokens = []
            code_tokens = []
            static_tags = []
            regex_tags = []
            ccg_parses = []
            score_shape = []
            verb_shape = []
            label = []
            it = 0
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(dataset))):
                    batch = dataset[i]
                    #                     if neg_sampling_strategy == 'random':
                    #                         neg_idx = sample_random(i, data)
                    #                     elif neg_sampling_strategy == 'hard':
                    #                         neg_idx, distances = sample_hard(i, distances)
                    sample = batch[shard_it]
                    try:
                        ccg_parse = sample[-1][1:-1]
                        tree = layout_net.construct_layout(ccg_parse)
                        tree = layout_net.remove_concats(tree)
                        code = sample[1]
                        scoring_inputs, verb_embeddings = layout_net.precompute_inputs(tree, code, [[], [], []],
                                                                                       [[], []], '')
                        scoring_outputs = layout_net.scoring_module.forward_batch(scoring_inputs[0], scoring_inputs[1])
                        verb_embeddings, _ = embedder.embed_batch(verb_embeddings[0], verb_embeddings[1])
                    except Exception as ex:
                        print(ex)
                        scoring_outputs = torch.FloatTensor(np.zeros((0, 512, 1)))
                        verb_embeddings = torch.FloatTensor(np.zeros((0, 1, 768)))

                    docstring_tokens.append(sample[0])
                    code_tokens.append(sample[1])
                    static_tags.append(sample[2])
                    regex_tags.append(sample[3])
                    ccg_parses.append(sample[4])
                    label.append(0)

                    new_scores_offset = scores_offset + scoring_outputs.shape[0]
                    score_shape.append((scores_offset, scoring_outputs.shape[0]))
                    scores_offsets_map[it] = scores_offset
                    scores_data_map[scores_offset:new_scores_offset] = scoring_outputs.cpu().numpy()
                    scores_offset = new_scores_offset

                    new_verbs_offset = verbs_offset + verb_embeddings.shape[0]
                    verb_shape.append((verbs_offset, verb_embeddings.shape[0]))
                    verbs_offsets_map[it] = verbs_offset
                    verbs_data_map[verbs_offset:new_verbs_offset] = verb_embeddings.cpu().numpy()
                    verbs_offset = new_verbs_offset

                    it += 1
                    scores_data_map.flush()
                    scores_offsets_map.flush()
                    verbs_data_map.flush()
                    verbs_offsets_map.flush()

                scores_offsets_map[it] = scores_offset
                scores_offsets_map.flush()
                verbs_offsets_map[it] = verbs_offset
                verbs_offsets_map.flush()
                new_df = pd.DataFrame(columns=['docstring_tokens', 'code_tokens', 'static_tags', 'regex_tags',
                                               'ccg_parse', 'score_shape', 'verb_shape', 'label'])
                new_df['docstring_tokens'] = docstring_tokens
                new_df['code_tokens'] = code_tokens
                new_df['static_tags'] = static_tags
                new_df['regex_tags'] = regex_tags
                new_df['ccg_parse'] = ccg_parses
                new_df['score_shape'] = score_shape
                new_df['verb_shape'] = verb_shape
                new_df['label'] = label

                new_df.to_json(data_dir + f'/ccg_train_{file_it}_{shard_it}.jsonl.gz', orient='records', lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute CodeBERT embeddings for data')
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on')
    parser.add_argument('--num_negatives', dest='num_negatives', type=int,
                        help='number of negative samples to include', required=True)
    parser.add_argument('--neg_sampling_strategy', dest='neg_sampling_strategy', type=str,
                        help='"random" or "hard" or "codebert"', required=True)
    parser.add_argument('--shard_size', dest='shard_size', type=int, default=30000)
    args = parser.parse_args()

    main(args.num_negatives, args.neg_sampling_strategy, args.shard_size)
