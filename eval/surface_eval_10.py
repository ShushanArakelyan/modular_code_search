import argparse
from datetime import datetime

import glob
import natsort
import numpy as np
import os
import torch
import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader

from eval.dataset import CodeSearchNetDataset, transform_sample
from eval.dataset import CodeSearchNetDataset_NotPrecomputed, CodeSearchNetDataset_TFIDFOracle
from eval.utils import mrr
from layout_assembly.layout import LayoutNet
from layout_assembly.layout_with_adapter import LayoutNetWithAdapters
from layout_assembly.modules import ScoringModule, ActionModuleFacade_v1, ActionModuleFacade_v2, ActionModuleFacade_v4
from layout_assembly.modules import ActionModuleFacade_v1_1_reduced, ActionModuleFacade_v2_1


device = 'cuda:0'
valid_file_name = '/home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl.gz' 
valid_dataset = CodeSearchNetDataset_NotPrecomputed(valid_file_name, device, neg_count=9)
valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)


def run_valid(data_loader, layout_net, count):
    MRRs = []
    with torch.no_grad(): 
        layout_net.precomputed_scores_provided = False 
        i = 0
        for samples in data_loader:
            if i == count:
                break
            i += 1
            ranks = []
            sample = samples[0]
            pred = layout_net.forward(*transform_sample(sample))
            if pred:
                ranks.append(torch.sigmoid(pred).cpu().numpy())
            else:
                continue
            for sample in samples[1:]:
                pred = layout_net.forward(*transform_sample(sample))
                if pred:
                    ranks.append(torch.sigmoid(pred).cpu().numpy())
                else:
                    ranks.append(np.random.rand(1)[0])
            MRRs.append(mrr(ranks))
        layout_net.precomputed_scores_provided = True
    return np.mean(MRRs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='End-to-end training of neural module network')
    parser.add_argument('--layoutnet_checkpoint', dest='model_checkpoint', type=str,
                        help='model checkpoint')
    parser.add_argument('--action_version', dest='action_version', type=int,
                        help='action version')
    args = parser.parse_args()
    
    scoring_checkpoint = '/home/shushan/finetuned_scoring_models/06-09-2021 20:23:12/model_3_ep_5.tar'
    scoring_module = ScoringModule(device, scoring_checkpoint)
    
    version = args.action_version
    if version == 1:
        action_module = ActionModuleFacade_v1(device)
    elif version == 2:
        action_module = ActionModuleFacade_v2(device)
    elif version == 4:
        action_module = ActionModuleFacade_v4(device)
    elif version == 11:
        action_module = ActionModuleFacade_v1_1_reduced(device)
    elif version == 21:
        action_module = ActionModuleFacade_v2_1(device)

    layout_net_version = 'classic'
    if layout_net_version == 'classic':
        layout_net = LayoutNet(scoring_module, action_module, device, precomputed_scores_provided=False)
    elif layout_net_version == 'with_adapters':
        layout_net = LayoutNetWithAdapters(scoring_module, action_module, device, precomputed_scores_provided=False)

    model_checkpoint = args.model_checkpoint
    checkpoint_dir = '/home/shushan/modular_code_search/model_checkpoints/action/' + model_checkpoint
    checkpoints = natsort.natsorted(glob.glob(checkpoint_dir + '/*'))

    with open("/home/shushan/eval_results/" + f'{model_checkpoint}_eval_on_10.csv', 'w') as f:
        for c in checkpoints:
            if c.endswith('.tar'):
                print("Evaluating checkpoint: ", c)
                layout_net.load_from_checkpoint(c)
                f.write(str(c.split('/')[-1]) + ', ' + str(run_valid(valid_data_loader, layout_net, 500)) + '\n')
                f.flush()