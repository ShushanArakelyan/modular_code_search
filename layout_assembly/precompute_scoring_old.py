import numpy as np
import pandas as pd
import torch
import tqdm

from layout_assembly.layout import LayoutNet
from layout_assembly.modules import ScoringModule, ActionModuleFacade_v1, ActionModuleFacade_v2

device = 'cuda:0'
data_file = '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_0.jsonl.gz'
scoring_checkpoint = '/home/shushan/finetuned_scoring_models/06-09-2021 20:23:12/model_3_ep_5.tar'
num_negatives = 1
new_df = pd.DataFrame(columns=['docstring_tokens', 'code_tokens', 'static_tags', 'regex_tags',
                               'ccg_parse', 'score_offsets', 'verb_offsets', 'code_offsets'])


def sample_hard(idx, data):
    pass


def sample_negative(idx, data):
    if NEG_SAMPLE_STRATEGY = 'random':
        np.random.seed(i)
        random_idx = np.random.randint(0, len(data), 1)[0]
        return random_idx
    elif NEG_SAMPLE_STRATEGY = 'hard':
        return sample_hard()


docstring_tokens = []
code_tokens = []
static_tags = []
regex_tags = []
ccg_parse = []
score_offsets = []
verb_offsets = []
code_offsets = []

for file_it in range(6):
    data_dir = '/home/shushan/train_v2'
    data_dir1 = '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train'
    data_file = f'{data_dir1}/ccg_train_{file_it}.jsonl.gz'
    data = pd.read_json(data_file, lines=True)
    scoring_module = ScoringModule(device, scoring_checkpoint)
    version = 2
    if version == 1:
        action_module = ActionModuleFacade_v1(device)
    elif version == 2:
        action_module = ActionModuleFacade_v2(device)
    layout_net = LayoutNet(scoring_module, action_module, device)
    loss_func = torch.nn.BCEWithLogitsLoss()
    op = torch.optim.Adam(layout_net.parameters(), lr=1e-4)

    positive = torch.FloatTensor([[1]]).to(device)
    negative = torch.FloatTensor([[0]]).to(device)

    scores_data_map = np.memmap(f'{data_dir}/memmap_scores_data_{file_it}.npy', dtype='float32', mode='w+',
                                shape=(200000, 512, 1))
    scores_offsets_map = np.memmap(f'{data_dir}/memmap_scores_offsets_{file_it}.npy', dtype='int32', mode='w+',
                                   shape=(60001, 1))

    verbs_data_map = np.memmap(f'{data_dir}/memmap_verbs_data_{file_it}.npy', dtype='float32', mode='w+',
                               shape=(200000, 1, 768))
    verbs_offsets_map = np.memmap(f'{data_dir}/memmap_verbs_offsets_{file_it}.npy', dtype='int32', mode='w+',
                                  shape=(60001, 1))

    code_data_map = np.memmap(f'{data_dir}/memmap_code_data_{file_it}.npy', dtype='float32', mode='w+',
                              shape=(200000, 512, 768))
    code_offsets_map = np.memmap(f'{data_dir}/memmap_code_offsets_{file_it}.npy', dtype='int32', mode='w+',
                                 shape=(60001, 1))

    scores_offset = 0
    verbs_offset = 0
    code_offset = 0
    new_scores_offset = 0
    new_verbs_offset = 0
    new_code_offset = 0
    it = 0
    with torch.no_grad():
        for li, label in enumerate([positive] + num_negative * [negative]):
            for i in tqdm.tqdm(range(len(data))):
                for param in layout_net.parameters():
                    param.grad = None
                ccg_parse = data['ccg_parse'][i][1:-1]
                if li == 0:
                    sample = (data['docstring_tokens'][i],
                              data['code_tokens'][i],
                              data['static_tags'][i],
                              data['regex_tags'][i],
                              data['ccg_parse'][i])
                else:
                    np.random.seed(i)
                    random_idx = np.random.randint(0, len(data), 1)[0]
                    sample = (data['docstring_tokens'][i],
                              data['code_tokens'][random_idx],
                              data['static_tags'][random_idx],
                              data['regex_tags'][random_idx],
                              data['ccg_parse'][i])
                try:
                    pred = layout_net.forward(ccg_parse, sample)
                    scoring_outputs = layout_net.scoring_outputs
                    verb_embeddings = layout_net.verb_embeddings
                    code_embeddings = layout_net.code_embeddings
                except:
                    scoring_outputs = torch.FloatTensor(np.zeros((0, 512, 1)))
                    verb_embeddings = torch.FloatTensor(np.zeros((0, 1, 768)))
                    code_embeddings = torch.FloatTensor(np.zeros((0, 512, 768)))

                new_scores_offset = scores_offset + scoring_outputs.shape[0]
                scores_offsets_map[it] = scores_offset
                scores_data_map[scores_offset:new_scores_offset] = scoring_outputs.cpu().numpy()
                scores_offset = new_scores_offset

                new_verbs_offset = verbs_offset + verb_embeddings.shape[0]
                verbs_offsets_map[it] = verbs_offset
                verbs_data_map[verbs_offset:new_verbs_offset] = verb_embeddings.cpu().numpy()
                verbs_offset = new_verbs_offset

                new_code_offset = code_offset + code_embeddings.shape[0]
                code_offsets_map[it] = code_offset
                code_data_map[code_offset:new_code_offset] = code_embeddings.cpu().numpy()
                code_offset = new_code_offset

                it += 1
                scores_data_map.flush()
                scores_offsets_map.flush()
                verbs_data_map.flush()
                verbs_offsets_map.flush()
                code_data_map.flush()
                code_offsets_map.flush()

        scores_offsets_map[it] = scores_offset
        scores_offsets_map.flush()
        verbs_offsets_map[it] = verbs_offset
        verbs_offsets_map.flush()
        code_offsets_map[it] = code_offset
        code_offsets_map.flush()


def main(num_negatives):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute CodeBERT embeddings for data')
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on')
    parser.add_argument('--num_negatives', dest='num_negatives', type=int,
                        help='number of negative samples to include', required=True)

    args = parser.parse_args()
    main(args.num_negatives)
