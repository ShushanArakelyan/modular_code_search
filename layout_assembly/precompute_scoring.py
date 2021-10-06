import argparse
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import euclidean_distances

import codebert_embedder as embedder
from layout_assembly.layout import LayoutNet
from layout_assembly.modules import ScoringModule, ActionModuleFacade

device = 'cuda:0'
scoring_checkpoint = "/home/shushan/finetuned_scoring_models/06-09-2021 20:21:51/model_3_ep_5.tar"


def sample_hard(idx, distances):
    neg_idx = np.argsort(distances[idx])[1]  # the zero-th element is idx itself
    distances[idx, neg_idx] = np.inf
    return neg_idx, distances


def sample_random(idx, data):
    random_idx = np.random.randint(0, len(data), 1)[0]
    return random_idx


def main(num_negatives, neg_sampling_strategy):
    for file_it in range(6):
        data_dir = f'/home/shushan/train_v2_neg_{num_negatives}_{neg_sampling_strategy}'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_dir1 = '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train'
        data_file = f'{data_dir1}/ccg_train_{file_it}.jsonl.gz'
        data = pd.read_json(data_file, lines=True)
        scoring_module = ScoringModule(device, scoring_checkpoint)
        version = 2
        action_module = ActionModuleFacade(device, version, normalized=False)
        layout_net = LayoutNet(scoring_module, action_module, device)

        positive = torch.FloatTensor([[1]]).to(device)
        negative = torch.FloatTensor([[0]]).to(device)

        offsets_count = int((num_negatives + 1) * len(data) + 1)
        data_map_count = int((num_negatives + 1) * len(data) * 3.5)

        if neg_sampling_strategy == 'hard':
            tfIdfVectorizer = TfidfVectorizer(use_idf=True)
            tfIdfVectorizer.fit_transform(
                [' '.join(dt) for dt in data['docstring_tokens']] + [' '.join(ct) for ct in data['code_tokens']])
            tfIdf_docs = tfIdfVectorizer.transform([' '.join(dt) for dt in data['docstring_tokens']])
            tfIdf_codes = tfIdfVectorizer.transform([' '.join(ct) for ct in data['code_tokens']])
            distances = euclidean_distances(tfIdf_docs, tfIdf_codes)

        scores_data_map = np.memmap(f'{data_dir}/memmap_scores_data_{file_it}.npy', dtype='float32', mode='w+',
                                    shape=(data_map_count, 512, 1))
        scores_offsets_map = np.memmap(f'{data_dir}/memmap_scores_offsets_{file_it}.npy', dtype='int32', mode='w+',
                                       shape=(offsets_count, 1))

        verbs_data_map = np.memmap(f'{data_dir}/memmap_verbs_data_{file_it}.npy', dtype='float32', mode='w+',
                                   shape=(data_map_count, 1, 768))
        verbs_offsets_map = np.memmap(f'{data_dir}/memmap_verbs_offsets_{file_it}.npy', dtype='int32', mode='w+',
                                      shape=(offsets_count, 1))

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
            for i in tqdm.tqdm(range(len(data))):
                np.random.seed(i)
                for li, label_i in enumerate([positive] + num_negatives * [negative]):
                    ccg_parse = data['ccg_parse'][i][1:-1]
                    if label_i == 1:
                        sample = (data['docstring_tokens'][i],
                                  data['alt_code_tokens'][i],
                                  data['static_tags'][i],
                                  data['regex_tags'][i],
                                  data['ccg_parse'][i])
                    else:
                        if neg_sampling_strategy == 'random':
                            neg_idx = sample_random(i, data)
                        elif neg_sampling_strategy == 'hard':
                            neg_idx, distances = sample_hard(i, distances)
                        sample = (data['docstring_tokens'][i],
                                  data['alt_code_tokens'][neg_idx],
                                  data['static_tags'][neg_idx],
                                  data['regex_tags'][neg_idx],
                                  data['ccg_parse'][i])
                    try:
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
                    label.append(int(label_i.data))

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

            new_df.to_json(data_dir + f'/ccg_train_{file_it}.jsonl.gz', orient='records', lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute CodeBERT embeddings for data')
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on')
    parser.add_argument('--num_negatives', dest='num_negatives', type=int,
                        help='number of negative samples to include', required=True)
    parser.add_argument('--neg_sampling_strategy', dest='neg_sampling_strategy', type=str,
                        help='"random" or "hard"', required=True)
    args = parser.parse_args()

    main(args.num_negatives, args.neg_sampling_strategy)
