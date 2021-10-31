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
from eval.dataset import CodeSearchNetDataset_wShards
from eval.utils import mrr
from layout_assembly.layout import LayoutNet
from layout_assembly.layout_with_adapter import LayoutNetWithAdapters
from layout_assembly.modules import ScoringModule, ActionModuleFacade
from layout_assembly.action_v1_w_entr_reg import ActionModule_v1_one_input, ActionModule_v1_two_inputs


class ActionModuleFacadeWEntrReg(ActionModuleFacade):
    def init_networks(self, version, normalized):
        if version == 1:
            self.one_input_module = ActionModule_v1_one_input(self.device, normalized, self.eval)
            self.two_inputs_module = ActionModule_v1_two_inputs(self.device, normalized, self.eval)
        else:
            raise Exception("Not implemented!")


def eval_mrr(data_loader, layout_net, count):
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


def eval_acc(data_loader, layout_net, count):
    accs = []
    with torch.no_grad():
        layout_net.precomputed_scores_provided = False
        i = 0
        for samples in data_loader:
            for j, sample in enumerate(samples):
                if i == count:
                    break
                i += 1
                pred = layout_net.forward(*transform_sample(sample))
                if pred:
                    if j == 0:
                        label = 1
                    else:
                        label = 0
                    accs.append(int(torch.sigmoid(pred).round() == label))
        layout_net.precomputed_scores_provided = True
    return np.mean(accs)


def main(device, data_dir, scoring_checkpoint, num_epochs, lr, print_every, save_every, version, layout_net_version,
         valid_file_name, num_negatives, precomputed_scores_provided, normalized_action, l1_reg_coef, adamw, example_count, layout_checkpoint=None):
    shard_range = num_negatives
    dataset = ConcatDataset([CodeSearchNetDataset_wShards(data_dir, r, shard_it, device) for r in range(1) for shard_it in
                             range(shard_range + 1)])

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    valid_dataset = CodeSearchNetDataset_SavedOracle(valid_file_name, device, neg_count=9,
                                                     oracle_idxs='/home/shushan/codebert_valid_oracle_scores_full.txt')
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    scoring_module = ScoringModule(device, scoring_checkpoint)
    action_module = ActionModuleFacadeWEntrReg(device, version, normalized_action)
    
    if layout_net_version == 'classic':
        layout_net = LayoutNet(scoring_module, action_module, device,
                               precomputed_scores_provided=precomputed_scores_provided)
    elif layout_net_version == 'with_adapters':
        layout_net = LayoutNetWithAdapters(scoring_module, action_module, device,
                                           precomputed_scores_provided=precomputed_scores_provided)
    if layout_checkpoint:
        layout_net.load_from_checkpoint(layout_checkpoint)
    
    loss_func = torch.nn.BCEWithLogitsLoss()
    op = torch.optim.Adam(layout_net.parameters(), lr=lr, weight_decay=adamw)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, verbose=True)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    writer = SummaryWriter(f'/home/shushan/modular_code_search/runs/{dt_string}')
    print("Writing to tensorboard: ", dt_string)
    writer_it = 0

    checkpoint_dir = f'/home/shushan/modular_code_search/model_checkpoints/action/{dt_string}'
    print("Checkpoints will be saved in ", checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    batch_size = 20
    for epoch in range(num_epochs):
        cumulative_loss = []
        accuracy = []
        checkpoint_prefix = checkpoint_dir + f'/model_{epoch}'
        loss = None
        steps = 0
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            if (steps + 1) % print_every == 0:
                writer.add_scalar("Loss/train",
                                  np.mean(cumulative_loss[-int(print_every/batch_size):]), writer_it)
                writer.add_scalar("Acc/train",
                                  np.mean(accuracy[-print_every:]), writer_it)
                writer.add_scalar("Acc/valid",
                                  np.mean(eval_acc(valid_data_loader, layout_net, count=50)), writer_it)
                scheduler.step(np.mean(cumulative_loss[-print_every:]))

            if (steps + 1) % save_every == 0:
                print("running validation evaluation....")
                writer.add_scalar("MRR/valid", eval_mrr(valid_data_loader, layout_net, count=500), writer_it)
                print("validation complete")
                print("saving to checkpoint: ")
                layout_net.save_to_checkpoint(checkpoint_prefix + f'_{i + 1}.tar')
                print("saved successfully")
                
            for param in layout_net.parameters():
                param.grad = None
            sample, scores, verbs, label = datum
            # data loader tries to put the scores, verbs and code_embeddings into a batch, thus
            # an extra dimension
            if scores[0].shape[0] == 0 or verbs[0].shape[0] == 0:
                continue
            layout_net.scoring_outputs = scores[0]
            pred = layout_net.forward(*transform_sample(sample))
            if pred is None:
                continue
            if loss is None:
                loss = loss_func(pred, label)
            else:
                loss += loss_func(pred, label)
            if l1_reg_coef > 0:
                for al in layout_net.accumulated_loss:
                    loss += l1_reg_coef * al
            steps += 1
            writer_it += 1  # this way the number in tensorboard will correspond to the actual number of iterations
            accuracy.append(int(torch.sigmoid(pred).round() == label))
            if steps % batch_size == 0:
                loss.backward()
                cumulative_loss.append(loss.data.cpu().numpy()/batch_size)
                op.step()
                loss = None
                for x in layout_net.parameters():
                    x.grad = None
            
            
            if steps >= example_count:
                break
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
                        help='print to tensorboard after this many iterations', default=1000)
    parser.add_argument('--save_every', dest='save_every', type=int,
                        help='save to checkpoint after this many iterations', default=50000)
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
    parser.add_argument('--num_negatives', dest='num_negatives', type=int,
                        help='Number of distractors to use in training')
    parser.add_argument('--precomputed_scores_provided', dest='precomputed_scores_provided',
                        default=False, action='store_true')
    parser.add_argument('--normalized_action', dest='normalized_action',
                        default=False, action='store_true')
    parser.add_argument('--l1_reg_coef', dest='l1_reg_coef', type=float,
                        default=0)
    parser.add_argument('--adamw', dest='adamw', type=float,
                        default=0)
    parser.add_argument('--example_count', dest='example_count', type=int,
                        default=0)

    args = parser.parse_args()
    main(args.device, args.data_dir, args.scoring_checkpoint, args.num_epochs, args.lr, args.print_every,
         args.save_every, args.version, args.layout_net_version, args.valid_file_name,
         args.num_negatives, args.precomputed_scores_provided, args.normalized_action, 
         args.l1_reg_coef, args.adamw, args.example_count, args.layout_checkpoint_file)
