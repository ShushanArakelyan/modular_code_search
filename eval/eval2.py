import argparse

import numpy as np
import pandas as pd
import torch

import codebert_embedder_v2 as embedder
from action2.action import ActionModule
from eval.utils import mrr
from layout_assembly.layout_ws2 import LayoutNetWS2 as LayoutNet
from utils import get_alignment_function
from layout_assembly.modules import ScoringModule

HOME_DIR = '/home/shushan'
DIR_NAME = f"{HOME_DIR}/codebert_1000/"
IS_VALID = False
IS_TEST = True

# create a map from url to data index
def create_map_from_url2idx(data):
    validation_data_map = {}
    for i in range(len(data)):
        validation_data_map[data['url'][i]] = i
    return validation_data_map

def p_at_k(rs, k):
    correct_score = rs[0]
    scores = np.array(rs[1:])
    rank = np.sum(scores >= correct_score) + 1
    if rank <= k:
        return 1
    else:
        return 0


def eval_modular(layout_net, data, make_prediction, idx, idxs_to_eval):
    ranks = []
    sample = (data['docstring_tokens'][idx],
              data['alt_code_tokens'][idx],
              data['static_tags'][idx],
              data['regex_tags'][idx],
              data['ccg_parse'][idx])
    try:
        pred = layout_net.forward(sample[-1][1:-1], sample)
        pred = make_prediction(pred)
        ranks.append(float(pred.cpu().numpy()))
        for neg_idx in idxs_to_eval:
            sample = (data['docstring_tokens'][idx],
                      data['alt_code_tokens'][neg_idx],
                      data['static_tags'][neg_idx],
                      data['regex_tags'][neg_idx],
                      data['ccg_parse'][idx])
            try:
                pred = layout_net.forward(sample[-1][1:-1], sample)
                pred = make_prediction(pred)
                ranks.append(float(pred.cpu().numpy()))
            except:
                np.random.seed(idx*neg_idx)
                ranks.append(np.random.rand(1)[0])
        return mrr(ranks), p_at_k(ranks, 1), p_at_k(ranks, 3), p_at_k(ranks, 5)
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description='End-to-end training of neural module network')
    parser.add_argument('--device', dest='device', type=str, help='device to run on')
    parser.add_argument('--model_name', dest='model_name', type=str, required=True)
    parser.add_argument('--top', dest='top', type=int, required=True)
    parser.add_argument('--alignment_func', dest='alignment_func', type=str, required=True)
    parser.add_argument('--home_dir', dest='home_dir', type=str)

    args = parser.parse_args()

    model_name = args.model_name
    device = args.device
    top = args.top
    alignment_function = args.alignment_func
    if args.home_dir:
        global HOME_DIR
        HOME_DIR = args.home_dir

    if IS_VALID and IS_TEST:
        raise Exception("Should be only one of two!")
    if IS_VALID:
        valid_file_name = f'{HOME_DIR}/CodeSearchNet/resources/ccg_parses_only_v2/python/final/jsonl/valid/ccg_python_valid_0.jsonl.gz'
    elif IS_TEST:
        valid_file_name = f'{HOME_DIR}/CodeSearchNet/resources/ccg_parses_only_v2/python/final/jsonl/test/ccg_python_test_0.jsonl.gz'
    else:
        raise Exception("Either is_valid or is_test should be set to true")
    data = pd.read_json(valid_file_name, lines=True)
    valid_data_map = create_map_from_url2idx()

    scoring_checkpoint = f"{HOME_DIR}/finetuned_scoring_models/03-01-2022 17:33:42/model_0_ep_03000.tar"
    scoring_module = ScoringModule(device, scoring_checkpoint)
    action_module = ActionModule(device, dim_size=embedder.dim, dropout=0.1)

    code_in_output, weighted_cosine, mlp_prediction, make_prediction = get_alignment_function(alignment_function)
    layout_net = LayoutNet(scoring_module, action_module, device, code_in_output, weighted_cosine)

    layout_net.finetune_scoring = True
    layout_net_training_ckp = f"{HOME_DIR}/modular_code_search/model_checkpoints/action/{model_name}/train/best_model.tar"
    layout_net.load_from_checkpoint(layout_net_training_ckp)

    codebert_mrr = []
    codebert_p_1 = []
    codebert_p_3 = []
    codebert_p_5 = []

    modular_mrr = []
    modular_p_1 = []
    modular_p_3 = []
    modular_p_5 = []

    eval_name = "test" if IS_TEST else "valid"
    with open(f"{HOME_DIR}/modular_code_search/eval_results/csn_{eval_name}_{model_name}_top{top}", 'w') as out_file:
        with torch.no_grad():
            for file_i in range(8):
                print(f"\n\n\nFile: {file_i}")
                filename = DIR_NAME + f"{file_i}_batch_result.txt"
                offset = file_i * 1000
                with open(filename, 'r') as f:
                    for j in range(1000):
                        scores = []
                        sample_idxs = []
                        for i in range(1000):
                            line = f.readline()
                            parts = line.split('<CODESPLIT>')
                            score = float(parts[-1].strip('\n'))
                            scores.append(score)
                            sample_idxs.append(valid_data_map[parts[2]])
                        scores = np.asarray(scores)
                        codebert_mrr.append(1. / (np.sum(scores >= scores[j])))
                        codebert_p_1.append(1 if np.sum(scores >= scores[j]) == 1 else 0)
                        codebert_p_3.append(1 if np.sum(scores >= scores[j]) <= 3 else 0)
                        codebert_p_5.append(1 if np.sum(scores >= scores[j]) <= 5 else 0)
                        idxs = np.argsort(scores)[::-1][:(top + 1)]
                        if j in idxs:
                            idxs = idxs[idxs != j]
                            idxs += offset
                            with torch.cuda.amp.autocast():
                                out = eval_modular(layout_net, data, make_prediction, j + offset, idxs[:top])
                            if out is None:
                                modular_mrr.append(codebert_mrr[-1])
                                modular_p_1.append(codebert_p_1[-1])
                                modular_p_3.append(codebert_p_3[-1])
                                modular_p_5.append(codebert_p_5[-1])
                            else:
                                mrr_, p_1, p_3, p_5 = out
                                modular_mrr.append(mrr_)
                                modular_p_1.append(p_1)
                                modular_p_3.append(p_3)
                                modular_p_5.append(p_5)
                                if codebert_mrr[-1] != mrr_:
                                    print(round(codebert_mrr[-1], 2), codebert_p_1[-1], codebert_p_3[-1],
                                          codebert_p_5[-1], str(round(mrr_, 2)), p_1, p_3, p_5)
                        else:
                            modular_mrr.append(codebert_mrr[-1])
                            modular_p_1.append(codebert_p_1[-1])
                            modular_p_3.append(codebert_p_3[-1])
                            modular_p_5.append(codebert_p_5[-1])
                        if (j + 1) % 50 == 0:
                            print(np.mean(modular_mrr), np.mean(modular_p_1), np.mean(modular_p_3),
                                  np.mean(modular_p_5), np.mean(codebert_mrr), np.mean(codebert_p_1),
                                  np.mean(codebert_p_3), np.mean(codebert_p_5))
                            out_file.write(
                                f"{str(np.mean(modular_mrr))} {str(np.mean(modular_p_1))} {str(np.mean(modular_p_3))} {str(np.mean(modular_p_5))} {str(np.mean(codebert_mrr))} {str(np.mean(codebert_p_1))} {str(np.mean(codebert_p_3))} {str(np.mean(codebert_p_5))}")
                            out_file.write('\n')
                            out_file.flush()
        out_file.write(
            f"{str(np.mean(modular_mrr))} {str(np.mean(modular_p_1))} {str(np.mean(modular_p_3))} {str(np.mean(modular_p_5))} {str(np.mean(codebert_mrr))} {str(np.mean(codebert_p_1))}  {str(np.mean(codebert_p_3))} {str(np.mean(codebert_p_5))}")
        out_file.write('\n')
        out_file.flush()

if __name__ == "__main__":
    main()