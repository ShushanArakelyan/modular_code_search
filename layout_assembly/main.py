import argparse
from datetime import datetime

import numpy as np
import os
import torch
import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader

from layout_assembly.layout import LayoutNet
from layout_assembly.layout_with_adapter import LayoutNetWithAdapters
from layout_assembly.modules import ScoringModule, ActionModuleFacade_v1, ActionModuleFacade_v2, ActionModuleFacade_v4
from layout_assembly.modules import ActionModuleFacade_v1_1_reduced
from eval.dataset import CodeSearchNetDataset, transform_sample


def main(device, data_dir, scoring_checkpoint, num_epochs, lr, print_every, save_every, version, layout_net_version):
    dataset = ConcatDataset([CodeSearchNetDataset(data_dir, r, device) for r in range(0, 1)])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    scoring_module = ScoringModule(device, scoring_checkpoint)
    if version == 1:
        action_module = ActionModuleFacade_v1(device)
    elif version == 2:
        action_module = ActionModuleFacade_v2(device)
    elif version == 4:
        action_module = ActionModuleFacade_v4(device)
    elif version == 11:
        action_module = ActionModuleFacade_v1_1_reduced(device)
    
    if layout_net_version == 'classic':
        layout_net = LayoutNet(scoring_module, action_module, device, precomputed_scores_provided=True)
    elif layout_net_version == 'with_adapters':
        layout_net = LayoutNetWithAdapters(scoring_module, action_module, device, precomputed_scores_provided=True)
    loss_func = torch.nn.BCEWithLogitsLoss()
    op = torch.optim.Adam(layout_net.parameters(), lr=lr)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    writer = SummaryWriter(f'/home/shushan/modular_code_search/runs/{dt_string}')
    print("Writing to tensorboard: ", dt_string)
    writer_it = 0
    
    checkpoint_dir = f'/home/shushan/modular_code_search/model_checkpoints/action/{dt_string}'
    print("Checkpoints will be saved in ", checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        cumulative_loss = []
        accuracy = []
        checkpoint_prefix = checkpoint_dir + f'/model_{epoch}'
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            for param in layout_net.parameters():
                param.grad = None
            sample, scores, verbs, code_embeddings, label = datum
            # data loader tries to put the scores, verbs and code_embeddings into a batch, thus
            # an extra dimension
            layout_net.scoring_outputs = scores[0]
            layout_net.verb_embeddings = verbs[0]
            layout_net.code_embeddings = code_embeddings[0]
            pred = layout_net.forward(*transform_sample(sample))
            if pred is None:
                continue
            loss = loss_func(pred, label)
            loss.backward()
            op.step()
            cumulative_loss.append(loss.data.cpu().numpy())
            accuracy.append(int(torch.sigmoid(pred).round() == label))
            del pred, loss
            if (i + 1) % print_every == 0:
                writer.add_scalar("Loss/train", 
                                  np.mean(cumulative_loss[-print_every:]), writer_it)
                writer.add_scalar("Acc/train", 
                                  np.mean(accuracy[-print_every:]), writer_it)
                writer_it += 1

            if (i + 1) % save_every == 0:
                print("saving to checkpoint: ")
                layout_net.save_to_checkpoint(checkpoint_prefix + f'_{i + 1}.tar')
                print("saved successfully")
        print("saving to checkpoint: ")
        layout_net.save_to_checkpoint(checkpoint_prefix + '.tar')
        print("saved successfully")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end training of neural module network')
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='training data directory', required=True)
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str,
                        help='Scoring module checkpoint', required=True)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        help='number of epochs to train', default=50)
    parser.add_argument('--print_every', dest='print_every', type=int,
                        help='print to tensorboard after this many iterations', default=100)
    parser.add_argument('--save_every', dest='save_every', type=int,
                        help='save to checkpoint after this many iterations', default=10000)
    parser.add_argument('--version', dest='version', type=int,
                        help='Whether to run ActionV1 or ActionV2', required=True)
    parser.add_argument('--lr', dest='lr', type=float,
                        help='learning rate', required=True)
    parser.add_argument('--layout_net_version', dest='layout_net_version', type=str,
                        help='"classic" or "with_adapters"', required=True)

    args = parser.parse_args()
    main(args.device, args.data_dir, args.scoring_checkpoint, args.num_epochs, args.lr, args.print_every, args.save_every, args.version, args.layout_net_version)
