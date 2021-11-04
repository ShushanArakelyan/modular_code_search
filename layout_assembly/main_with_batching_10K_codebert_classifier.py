import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval.dataset import CodeSearchNetDataset_wShards
from eval.dataset import transform_sample
from eval.utils import mrr
from layout_assembly.action_v1_codebert_classifier import ActionModule_v1_one_input, ActionModule_v1_two_inputs
from layout_assembly.layout_codebert_classifier import LayoutNet_w_codebert_classifier as LayoutNet
from layout_assembly.layout_with_adapter import LayoutNetWithAdapters
from layout_assembly.modules import ScoringModule, ActionModuleFacade


class ActionModuleFacade_w_codebert_classifier(ActionModuleFacade):
    def init_networks(self, version, normalized):
        super().init_networks(version, normalized)
        if version == 1:
            self.one_input_module = ActionModule_v1_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v1_two_inputs(self.device, normalized, self.dropout)


def eval_modular(data, idx, idxs_to_eval, layout_net):
    ranks = []
    sample = (data['docstring_tokens'][idx],
              data['alt_code_tokens'][idx],
              data['static_tags'][idx],
              data['regex_tags'][idx],
              data['ccg_parse'][idx])
    pred = layout_net.forward(sample[-1][1:-1], sample)
    if pred is None:
        return None
    else:
        ranks.append(float(pred[0][1].cpu().numpy()))
        for neg_idx in idxs_to_eval:
            sample = (data['docstring_tokens'][idx],
                      data['alt_code_tokens'][neg_idx],
                      data['static_tags'][neg_idx],
                      data['regex_tags'][neg_idx],
                      data['ccg_parse'][idx])
            pred = layout_net.forward(sample[-1][1:-1], sample)
            if pred is not None:
                ranks.append(float(pred[0][1].cpu().numpy()))
            else:
                np.random.seed(neg_idx)
                ranks.append(np.random.rand(1)[0])
    return mrr(ranks)


def eval_mrr(data, data_dir_name, data_map, layout_net):
    codebert_mrr = []
    modular_mrr = []
    with torch.no_grad():
        for file_i in range(8):
            filename = data_dir_name + f"{file_i}_batch_result.txt"
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
                        sample_idxs.append(data_map[parts[2]])
                    scores = np.asarray(scores)
                    codebert_mrr.append(1. / (np.sum(scores >= scores[j])))
                    idxs = np.argsort(scores)[::-1][:10]
                    if j in idxs:
                        idxs = idxs[idxs != j]
                        idxs += offset
                        out = eval_modular(data, j, idxs[:9], layout_net)
                        if out is None:
                            modular_mrr.append(codebert_mrr[-1])
                        else:
                            modular_mrr.append(out)
                    else:
                        modular_mrr.append(codebert_mrr[-1])
                    if (j + 1) % 50 == 0:
                        return np.mean(modular_mrr)


def eval_acc(data_loader, layout_net, count):
    accs = []
    layout_net.set_eval()
    with torch.no_grad():
        layout_net.precomputed_scores_provided = False
        i = 0
        for sample in data_loader:
            sample, scores, verbs, label = sample
            if i == count:
                break
            i += 1
            pred = layout_net.forward(*transform_sample(sample))
            if pred is None:
                continue
            accs.append(int(torch.argmax(pred) == label))
        layout_net.precomputed_scores_provided = True
    layout_net.set_train()
    return np.mean(accs)


def main(device, data_dir, scoring_checkpoint, num_epochs, lr, print_every, save_every, version, layout_net_version,
         valid_file_name, num_negatives, precomputed_scores_provided, normalized_action, l1_reg_coef, adamw,
         example_count, dropout, load_finetuned_codebert, checkpoint_dir, summary_writer_dir,
         codebert_valid_results_dir, use_lr_scheduler, clip_grad_value, layout_checkpoint=None):
    shard_range = num_negatives
    dataset = ConcatDataset(
        [CodeSearchNetDataset_wShards(data_dir, r, shard_it, device) for r in range(1) for shard_it in
         range(shard_range + 1)])

    if valid_file_name != "None":
        valid_data = pd.read_json(valid_file_name, lines=True)
        validation_data_map = {}
        for i in range(len(valid_data)):
            validation_data_map[valid_data['url'][i]] = i
    else:
        valid_dataset, dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.3), int(len(dataset)*0.7)], 
                                              generator=torch.Generator().manual_seed(42))
        valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    if load_finetuned_codebert:
        import codebert_embedder_v2 as embedder
        embedder.init_embedder(device, load_finetuned_codebert)

    scoring_module = ScoringModule(device, scoring_checkpoint)
    action_module = ActionModuleFacade_w_codebert_classifier(device, version, normalized_action, dropout)

    if layout_net_version == 'classic':
        if version == 5 or version == 6:
            return_separators = True
        else:
            return_separators = False
        if version == 7:
            embed_in_list = True
        else:
            embed_in_list = False
        layout_net = LayoutNet(scoring_module, action_module, device,
                               precomputed_scores_provided=precomputed_scores_provided,
                               return_separators=return_separators, embed_in_list=embed_in_list)
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
    writer = SummaryWriter(summary_writer_dir + f'/{dt_string}')
    print("Writing to tensorboard: ", dt_string)
    writer_it = 0

    checkpoint_dir = checkpoint_dir + f'/{dt_string}'
    print("Checkpoints will be saved in ", checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    positive_label = torch.FloatTensor([[0, 1]]).to(device)
    negative_label = torch.FloatTensor([[1, 0]]).to(device)
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
                                  np.mean(cumulative_loss[-int(print_every / batch_size):]), writer_it)
                writer.add_scalar("Acc/train",
                                  np.mean(accuracy[-print_every:]), writer_it)
                layout_net.set_eval()
                if valid_file_name != "None":
                    writer.add_scalar("MRR/valid",
                                      eval_mrr(valid_data, codebert_valid_results_dir, validation_data_map, layout_net),
                                      writer_it)
                else:
                    writer.add_scalar("Acc/valid", eval_acc(valid_data_loader, layout_net, count=100), writer_it)
                layout_net.set_train()
                if use_lr_scheduler:
                    scheduler.step(np.mean(cumulative_loss[-print_every:]))

            if (steps + 1) % save_every == 0:
                print("saving to checkpoint: ")
                layout_net.save_to_checkpoint(checkpoint_prefix + f'_{i + 1}.tar')
                print("saved successfully")

            for param in layout_net.parameters():
                param.grad = None
            sample, scores, verbs, label = datum
            if int(label) == 0:
                label = negative_label
            else:
                label = positive_label
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
            accuracy.append(int(torch.argmax(torch.sigmoid(pred)) == torch.argmax(label)))
            if steps % batch_size == 0:
                loss.backward()
                cumulative_loss.append(loss.data.cpu().numpy() / batch_size)
                if clip_grad_value > 0:
                    torch.nn.utils.clip_grad_value_(layout_net.parameters(), clip_grad_value)
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
    parser.add_argument('--device', dest='device', type=str, help='device to run on')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='training data directory', required=True)
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str,
                        help='Scoring module checkpoint', required=True)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, help='number of epochs to train', default=50)
    parser.add_argument('--print_every', dest='print_every', type=int,
                        help='print to tensorboard after this many iterations', default=1000)
    parser.add_argument('--save_every', dest='save_every', type=int,
                        help='save to checkpoint after this many iterations', default=50000)
    parser.add_argument('--version', dest='version', type=int,
                        help='Whether to run ActionV1 or ActionV2', required=True)
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate', required=True)
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
    parser.add_argument('--normalized_action', dest='normalized_action', default=False, action='store_true')
    parser.add_argument('--l1_reg_coef', dest='l1_reg_coef', type=float, default=0)
    parser.add_argument('--adamw', dest='adamw', type=float, default=0)
    parser.add_argument('--example_count', dest='example_count', type=int, default=0)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str)
    parser.add_argument('--summary_writer_dir', dest='summary_writer_dir', type=str)
    parser.add_argument('--codebert_valid_results_dir', dest='codebert_valid_results_dir', type=str)
    parser.add_argument('--load_finetuned_codebert', dest='load_finetuned_codebert', default=False, action='store_true')
    parser.add_argument('--use_lr_scheduler', dest='use_lr_scheduler', default=False, action='store_true')
    parser.add_argument('--clip_grad_value', dest='clip_grad_value', default=0, type=float)

    args = parser.parse_args()
    main(device=args.device,
         data_dir=args.data_dir,
         scoring_checkpoint=args.scoring_checkpoint,
         num_epochs=args.num_epochs,
         lr=args.lr,
         print_every=args.print_every,
         save_every=args.save_every,
         version=args.version,
         layout_net_version=args.layout_net_version,
         valid_file_name=args.valid_file_name,
         num_negatives=args.num_negatives,
         precomputed_scores_provided=args.precomputed_scores_provided,
         normalized_action=args.normalized_action,
         l1_reg_coef=args.l1_reg_coef,
         adamw=args.adamw,
         example_count=args.example_count,
         load_finetuned_codebert=args.load_finetuned_codebert,
         dropout=args.dropout,
         checkpoint_dir=args.checkpoint_dir,
         summary_writer_dir=args.summary_writer_dir,
         codebert_valid_results_dir=args.codebert_valid_results_dir,
         use_lr_scheduler=args.use_lr_scheduler,
         clip_grad_value=args.clip_grad_value,
         layout_checkpoint=args.layout_checkpoint_file)
