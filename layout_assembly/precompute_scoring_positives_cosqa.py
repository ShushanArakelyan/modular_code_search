import argparse
import os

import numpy as np
import pandas as pd
import torch
import tqdm

import codebert_embedder as embedder
from layout_assembly.layout import LayoutNet
from layout_assembly.modules import ScoringModule, ActionModuleFacade

device = 'cuda:0'


def main(shard_size, scoring_checkpoint):
    file_it = 0
    data_dir = '/project/hauserc_374/shushan/train_v2_cosqa_positive'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_file = '/project/hauserc_374/shushan/CoSQA/resources/ccg_parses_only/python/final/jsonl/train/ccg_cosqa_parsed.jsonl.gz'
    data = pd.read_json(data_file, lines=True)
    scoring_module = ScoringModule(device, scoring_checkpoint)
    version = 1
    action_module = ActionModuleFacade(device, version, normalized=False)
    layout_net = LayoutNet(scoring_module, action_module, device)

    offsets_count = int(shard_size + 1)
    data_map_count = int(shard_size * 3.5)

    for shard_it in range(1):
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
            for idx in tqdm.tqdm(range(len(data))):
                sample = (data['docstring_tokens'][idx],
                          data['alt_code_tokens'][idx],
                          data['static_tags'][idx],
                          data['regex_tags'][idx],
                          data['ccg_parse'][idx])
                try:
                    ccg_parse = sample[-1][1:-1]
                    tree = layout_net.construct_layout(ccg_parse)
                    tree = layout_net.remove_concats(tree)
                    code = sample[1]
                    scoring_inputs, verb_embeddings = layout_net.precompute_inputs(tree, code, [[], [], []],
                                                                                   [[], []], '')
                    scoring_outputs = layout_net.scoring_module.forward_batch_no_grad(scoring_inputs[0], scoring_inputs[1])
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
                label.append(1)

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
    parser.add_argument('--shard_size', dest='shard_size', type=int, default=30000)
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str, required=True)
    args = parser.parse_args()

    main(shard_size=args.shard_size, scoring_checkpoint=args.scoring_checkpoint)
