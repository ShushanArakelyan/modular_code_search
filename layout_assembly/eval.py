import pandas as pd
import numpy as np
import tqdm
from datetime import datetime

import torch

from torch.utils.tensorboard import SummaryWriter
from layout_assembly.layout import LayoutNet
from layout_assembly.modules import ScoringModule, ActionModuleFacade_v1


def mrr(rs):
    correct_score = rs[0]
    scores = np.array(rs[1:])
    rank = np.sum(scores >= correct_score) + 1
    return np.mean(1.0 / rank)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune codebert model for attend module')
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str,
                        help='load scoring model saved in checkpoint', required=True)
    parser.add_argument('--action_checkpoint', dest='action_checkpoint', type=str,
                        help='load action model saved in checkpoint', required=True)
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on', required=True)
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                            help='validation file name', required=True)

    args = parser.parse_args()
        
    data = pd.read_json(args.valid_file_name, lines=True)
    device = args.device
    
    scoring_module = ScoringModule(device, checkpoint=args.scoring_checkpoint, eval=True)
    action_module = ActionModuleFacade_v1(device, checkpoint=args.action_checkpoint, eval=True)
    layout_net = LayoutNet(scoring_module, action_module, device, eval=True)
    
    
    MRRs = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(data))):
            ranks = []
            ccg_parse = data['ccg_parse'][i][1:-1]
            sample = (data['docstring_tokens'][i], 
                 data['code_tokens'][i], 
                 data['static_tags'][i], 
                 data['regex_tags'][i], 
                 data['ccg_parse'][i])
            pred = layout_net.forward(ccg_parse, sample)
            if pred:
                ranks.append(torch.sigmoid(pred).cpu().numpy())
            else:
                ranks.append(np.random.rand(1)[0])
            np.random.seed(i)
            for j in tqdm.tqdm(range(999)):
                random_idx = np.random.randint(0, len(data), 1)[0]         
                sample = (data['docstring_tokens'][i], 
                         data['code_tokens'][random_idx], 
                         data['static_tags'][random_idx], 
                         data['regex_tags'][random_idx], 
                         data['ccg_parse'][i])
                pred = layout_net.forward(ccg_parse, sample)
                if pred:
                    ranks.append(torch.sigmoid(pred).cpu().numpy())
                else:
                    ranks.append(np.random.rand(1)[0])
            MRRs.append(mrr(ranks))
        print(np.mean(MRR))
