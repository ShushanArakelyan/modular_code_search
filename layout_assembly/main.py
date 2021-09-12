import argparse
import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from layout_assembly.layout import LayoutNet
from layout_assembly.modules import ScoringModule, ActionModuleFacade_v1


def main():
    parser = argparse.ArgumentParser(description='End-to-end training of neural module network')
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on')
    parser.add_argument('--data_file', dest='data_file', type=str,
                        help='training data directory', required=True)
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str,
                        help='Scoring module checkpoint', required=True)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        help='number of epochs to train', default=10)
    parser.add_argument('--print_every', dest='print_every', type=int,
                        help='number of epochs to train', default=100)

    args = parser.parse_args()
    device = args.device
    data = pd.read_json(args.data_file, lines=True)
    scoring_module = ScoringModule(device, args.scoring_checkpoint)
    action_module = ActionModuleFacade_v1(device)
    layout_net = LayoutNet(scoring_module, action_module, device)
    loss_func = torch.nn.BCEWithLogitsLoss()
    op = torch.optim.Adam(action_module.parameters(), lr=1e-5)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    writer = SummaryWriter(f'/home/shushan/modular_code_search/runs/{dt_string}')
    print("Writing to tensorboard: ", dt_string)
    
    print_every = args.print_every
    writer_it = 0
    positive = torch.FloatTensor([[1]]).to(device)
    negative = torch.FloatTensor([[0]]).to(device)
    for _ in args.num_epochs:
        cumulative_loss = []
        for i in tqdm.tqdm(range(len(data))):
            for li, label in enumerate([positive, negative]):
                op.zero_grad()
                ccg_parse = data['ccg_parse'][i][1:-1]
                if li == 0 :                
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
                pred = layout_net.forward(ccg_parse, sample)
                if pred is None:
                    f.write(f'{i}\n')
                    f.flush()
                    continue
                loss = loss_func(pred, label)
                loss.backward()
                op.step()
                cumulative_loss.append(loss.data.cpu().numpy())
            if i % print_every == 0:
                writer.add_scalar("Loss/train", np.mean(cumulative_loss[-print_every:]), writer_it)
                writer_it += 1


if __name__ == '__main__':
    main()
