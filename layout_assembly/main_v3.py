import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from layout_assembly.layout_v2 import LayoutNet_v2 as LayoutNet
from layout_assembly.modules import ScoringModule, ActionModuleFacade_v3


def main(device, data_file, scoring_checkpoint, print_every, save_every, num_epochs):
    
    data = pd.read_json(data_file, lines=True)
    scoring_module = ScoringModule(device, scoring_checkpoint)
    # version = args.version
    action_module = ActionModuleFacade_v3(device)
    # if version == 1:
    #     action_module = ActionModuleFacade_v1(device)
    # elif version == 2:
    #     action_module = ActionModuleFacade_v2(device)
    layout_net = LayoutNet(scoring_module, action_module, device)
    loss_func = torch.nn.BCEWithLogitsLoss()
    op = torch.optim.Adam(layout_net.parameters(), lr=1e-4)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    writer = SummaryWriter(f'/home/shushan/modular_code_search/runs/{dt_string}')
    print("Writing to tensorboard: ", dt_string)

    writer_it = 0
    positive = torch.FloatTensor([[1]]).to(device)
    negative = torch.FloatTensor([[0]]).to(device)
    for _ in range(num_epochs):
        cumulative_loss = []
        accuracy = []
        for i in tqdm.tqdm(range(len(data))):
            for li, label in enumerate([positive, negative]):
                op.zero_grad()
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
                pred = layout_net.forward(ccg_parse, sample)
                if pred is None:
                    continue
                loss = loss_func(pred, label)
                loss.backward()
                op.step()
                cumulative_loss.append(loss.data.cpu().numpy())
                accuracy.append(int(torch.sigmoid(pred).round() == label))
            if (i + 1) % print_every == 0:
                writer.add_scalar("Loss/train", np.mean(cumulative_loss[-print_every:]), writer_it)
                writer.add_scalar("Acc/train", np.mean(accuracy[-print_every:]), writer_it)
                writer_it += 1

            if (i + 1) % save_every == 0:
                print("saving to checkpoint: ")
                layout_net.save_to_checkpoint(f"/home/shushan/action_test_checkpoint_v_3_it_{i + 1}")
                print("saved successfully")


if __name__ == '__main__':
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
                        help='print to tensorboard after this many iterations', default=100)
    parser.add_argument('--save_every', dest='save_every', type=int,
                        help='save to checkpoint after this many iterations', default=5000)
    # parser.add_argument('--version', dest='version', type=int,
    #                     help='Whether to run ActionV1 or ActionV2', required=True)

    args = parser.parse_args()
    device = args.device
    data_file = args.data_file
    scoring_checkpoint = args.scoring_checkpoint
    print_every = args.print_every
    save_every = args.save_every
    num_epochs = args.num_epochs
    main(device, data_file, scoring_checkpoint, print_every, save_every, num_epochs)
