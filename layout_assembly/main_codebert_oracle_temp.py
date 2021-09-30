import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval.dataset import CodeSearchNetDataset, transform_sample
from eval.dataset import CodeSearchNetDataset_SavedOracle
from eval.utils import mrr
from layout_assembly.layout import LayoutNet
from layout_assembly.layout_with_adapter import LayoutNetWithAdapters
from layout_assembly.modules import ActionModuleFacade_v1_1_reduced, ActionModuleFacade_v2_1
from layout_assembly.modules import ScoringModule, ActionModuleFacade_v1, ActionModuleFacade_v2, ActionModuleFacade_v4


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


def main(device, data_dir, scoring_checkpoint, num_epochs, lr, print_every, save_every, version, layout_net_version,
         valid_file_name, oracle_negatives_dir, num_negatives, layout_checkpoint=None):
    if '_neg_10_' in data_dir:  # ugly, ugly, ugly
        dataset = ConcatDataset([CodeSearchNetDataset(data_dir, r, device) for r in range(0, 3)])
    else:
        if oracle_negatives_dir is None:
            dataset = ConcatDataset([CodeSearchNetDataset(data_dir, r, device) for r in range(0, 1)])
        else:
            dataset = CodeSearchNetDataset_SavedOracle(data_dir, device,
                                                       oracle_idxs=oracle_negatives_dir,
                                                       neg_count=num_negatives)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    valid_dataset = CodeSearchNetDataset_SavedOracle(valid_file_name, device, neg_count=9,
                                                     oracle_idxs='/home/shushan/codebert_valid_oracle_scores.txt')
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    scoring_module = ScoringModule(device, scoring_checkpoint)
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

    if layout_net_version == 'classic':
        layout_net = LayoutNet(scoring_module, action_module, device, precomputed_scores_provided=False)
    elif layout_net_version == 'with_adapters':
        layout_net = LayoutNetWithAdapters(scoring_module, action_module, device, precomputed_scores_provided=False)
    if layout_checkpoint:
        layout_net.load_from_checkpoint(layout_checkpoint)
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
        for i, samples in tqdm.tqdm(enumerate(data_loader)):
            for param in layout_net.parameters():
                param.grad = None
            sample = samples[0]
            pred = layout_net.forward(*transform_sample(sample))
            if pred is None:
                continue
            loss = loss_func(pred, dataset.positive_label.unsqueeze(dim=0))
            loss.backward()
            accuracy.append(int(torch.sigmoid(pred).round() == dataset.positive_label))
            for sample in samples[1:]:
                pred = layout_net.forward(*transform_sample(sample))
                if pred is None:
                    continue
                loss = loss_func(pred, dataset.negative_label.unsqueeze(dim=0))
                loss.backward()
                accuracy.append(int(torch.sigmoid(pred).round() == dataset.negative_label))
            op.step()
            cumulative_loss.append(loss.data.cpu().numpy())
            del pred, loss
            if (i + 1) % print_every == 0:
                writer.add_scalar("Loss/train",
                                  np.mean(cumulative_loss[-print_every:]), writer_it)
                writer.add_scalar("Acc/train",
                                  np.mean(accuracy[-print_every:]), writer_it)
                writer_it += 1

            if (i + 1) % save_every == 0:
                print("running validation evaluation....")
                mrr = run_valid(valid_data_loader, layout_net, count=500)
                writer.add_scalar("MRR/valid", mrr, writer_it)
                print("validation complete")
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
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='Validation data file', required=True)
    parser.add_argument('--layout_checkpoint_file', dest='layout_checkpoint_file', type=str,
                        help='Continue training from this checkpoint, not implemented')
    parser.add_argument('--oracle_negatives_dir', dest='oracle_negatives_dir', type=str,
                        help='Directory with distractors by oracle')
    parser.add_argument('--num_negatives', dest='num_negatives', type=int,
                        help='Number of distractors to use in training')

    args = parser.parse_args()
    main(args.device, args.data_dir, args.scoring_checkpoint, args.num_epochs, args.lr, args.print_every,
         args.save_every, args.version, args.layout_net_version, args.valid_file_name,
         args.oracle_negatives_dir, args.num_negatives, args.layout_checkpoint_file)
