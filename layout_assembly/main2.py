import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import codebert_embedder_v2 as embedder
from action2.action import ActionModule
from eval.dataset import CodeSearchNetDataset_wShards, CodeSearchNetDataset_NotPrecomputed
from eval.dataset import transform_sample, filter_neg_samples
from eval.utils import mrr, p_at_k
from layout_assembly.layout_ws2 import LayoutNetWS2 as LayoutNet
from layout_assembly.modules import ScoringModule
from layout_assembly.utils import ProcessingException

from sklearn.metrics import f1_score

def create_neg_sample(orig, distr):
    return (orig[0], distr[1], distr[2], distr[3], orig[4])


def binarize(a):
    with torch.no_grad():
        return torch.where(a < 0.5, torch.zeros_like(a), torch.ones_like(a))


def compute_alignment(a, b):
    return torch.sigmoid(torch.dot(a, b))


def make_prediction(output_list, device):
    output_tensor = torch.cat(output_list, dim=0)
    print("output_tensor: ", output_tensor.shape)
    output_tensor = output_tensor.squeeze()
    print("output_tensor: ", output_tensor.shape)
    alignment_scores = torch.sigmoid(torch.dot(output_tensor[:, 0], output_tensor[:, 1]))
    print("alignment_scores: ", alignment_scores.shape)
    pred = torch.prod(alignment_scores)
    print("pred: ", pred.shape)
    return pred


def eval_mrr_and_p_at_k(dataset, layout_net, device, k=[1], distractor_set_size=100, count=250):
    def get_mrr_for_one_sample(dataset, idx, idxs_to_eval, layout_net, k):
        ranks = []
        sample, _, _, _ = dataset[idx]
        try:
            output_list = layout_net.forward(sample[-1][1:-1], sample)
            pred = make_prediction(output_list, device)
        except ProcessingException:
            return None, None
        ranks.append(float(torch.sigmoid(pred).cpu().numpy()))
        for neg_idx in idxs_to_eval:
            distractor, _, _, _ = dataset[neg_idx]
            neg_sample = create_neg_sample(sample, distractor)
            try:
                output_list = layout_net.forward(neg_sample[-1][1:-1], neg_sample)
                pred = make_prediction(output_list, device)
                ranks.append(float(torch.sigmoid(pred).cpu().numpy()))
            except ProcessingException:
                np.random.seed(neg_idx)
                ranks.append(np.random.rand(1)[0])
        return mrr(ranks), [p_at_k(ranks, ki) for ki in k]
    results = {f'P@{ki}': [] for ki in k}
    results['MRR'] = []
    with torch.no_grad():
        np.random.seed(123)
        examples = np.random.choice(range(len(dataset)), count, replace=False)
        for ex in examples:
            np.random.seed(ex)
            idxs = np.random.choice(range(len(dataset)), distractor_set_size, replace=False)
            cur_mrr, p_at_ks = get_mrr_for_one_sample(dataset, ex, idxs, layout_net, k)
            if cur_mrr is None or p_at_ks is None:
                continue
            results['MRR'].append(cur_mrr)
            for ki, pre in zip(k, p_at_ks):
                results[f'P@{ki}'].append(pre)
    return np.mean(results['MRR']), [np.mean(results[f'P@{ki}']) for ki in k]


def eval_acc(dataset, layout_net, count, device):
    def get_acc_for_one_sample(sample, label):
        output_list = layout_net.forward(sample[-1][1:-1], sample)
        pred = make_prediction(output_list, device)
        binarized_pred = binarize(torch.sigmoid(pred))
        return int(binarized_pred == label)

    accs = []
    layout_net.set_eval()
    with torch.no_grad():
        i = 0
        for sample in range(len(dataset)):
            sample, _, _, label = dataset[i]
            assert label == 1, 'Mismatching example sampled from dataset, but expected matching examples only'
            try:
                accs.append(get_acc_for_one_sample(sample, label))
            except ProcessingException:
                continue
            # Create a negative example
            np.random.seed(22222 + i)
            neg_idx = np.random.choice(range(len(dataset)), 1)[0]
            neg_sample = create_neg_sample(dataset[i][0], dataset[neg_idx][0])
            try:
                accs.append(get_acc_for_one_sample(neg_sample, label=0))
            except ProcessingException:
                continue
            if i >= count:
                break
            i += 1
    layout_net.set_train()
    return np.mean(accs)


def pretrain(layout_net, lr, adamw, checkpoint_dir, num_epochs, data_loader, clip_grad_value, example_count, device,
             print_every, writer, k, valid_data, distractor_set_size, patience, use_lr_scheduler, batch_size):
    loss_func = torch.nn.BCEWithLogitsLoss()
    op = torch.optim.Adam(layout_net.parameters(), lr=lr, weight_decay=adamw)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, verbose=True)
    layout_net.set_train()
    checkpoint_dir += '/pretrain'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer_it = 0
    best_accuracy = -1.0
    wait_step = 0
    stop_training = False

    for epoch in range(num_epochs):
        cumulative_loss = []
        accuracy = []
        f1 = []
        loss = None
        steps = 0
        for x in layout_net.parameters():
            x.grad = None
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            if stop_training:
                break
            sample, _, _, _ = datum
            try:
                output_list = layout_net.forward(*transform_sample(sample))
            except ProcessingException:
                continue
            for out in output_list:
                true_out, pred_out = out
                labels = binarize(true_out).to(device)
                l = loss_func(pred_out, labels)
                if loss is None:
                    if torch.isnan(l).data:
                        print("Stop pretraining because loss=%s" % (l.data))
                        stop_training = True
                        break
                    loss = l
                else:
                    if torch.isnan(l).data:
                        print("Stop pretraining because loss=%s" % (l.data))
                        stop_training = True
                        break
                    loss += l

                binarized_preds = binarize(torch.sigmoid(pred_out))
                accuracy.append(sum((binarized_preds == labels).cpu().detach().numpy()) * 1. / labels.shape[0])
                f1.append(f1_score(labels.cpu().detach().numpy().flatten(),
                                   binarized_preds.cpu().detach().numpy().flatten(), zero_division=1))

            steps += 1
            writer_it += 1  # this way the number in tensorboard will correspond to the actual number of iterations
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
            if steps % print_every == 0:
                writer.add_scalar("Pretraining Loss/pretraining",
                                  np.mean(cumulative_loss[-int(print_every / batch_size):]), writer_it)
                writer.add_scalar("Pretraining Acc/pretraining",
                                  np.mean(accuracy[-print_every:]), writer_it)
                writer.add_scalar("Pretraining F1/pretraining",
                                  np.mean(f1[-print_every:]), writer_it)
                layout_net.set_eval()
                acc = eval_acc(valid_data, layout_net, count=1000, device=device)
                writer.add_scalar("Pretraining Acc/inference", acc, writer_it)
                cur_perf = acc
                print("Best pretraining performance: ", best_accuracy)
                print("Current pretraining performance: ", cur_perf)
                print("best < current: ", best_accuracy < cur_perf)
                if best_accuracy < cur_perf:
                    layout_net.save_to_checkpoint(checkpoint_dir + '/best_model.tar')
                    print(
                        "Saving model with best pretraining accuracy performance: %s -> %s on epoch=%d, global_step=%d" %
                        (best_accuracy, cur_perf, epoch, steps))
                    best_accuracy = cur_perf
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= patience:
                        print("Stopping pretraining because wait steps exceeded: ", wait_step)
                        stop_training = True
                layout_net.set_train()
                if use_lr_scheduler:
                    scheduler.step(np.mean(cumulative_loss[-print_every:]))


def train(device, layout_net, lr, adamw, checkpoint_dir, num_epochs, data_loader, clip_grad_value, example_count,
          use_lr_scheduler, writer, valid_data, k, distractor_set_size, print_every, patience, batch_size, finetune_scoring):
    loss_func = torch.nn.BCEWithLogitsLoss()
    op = torch.optim.Adam(layout_net.parameters(), lr=lr, weight_decay=adamw)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, verbose=True)
    checkpoint_dir += '/train'
    # if finetune_scoring:
    #     layout_net.finetune_scoring = True
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    positive_label = torch.tensor(1, dtype=float).to(device)
    negative_label = torch.tensor(0, dtype=float).to(device)

    writer_it = 0
    best_accuracy = (-1.0, -1.0, -1.0)
    wait_step = 0
    stop_training = False

    for epoch in range(num_epochs):
        cumulative_loss = []
        accuracy = []
        loss = None
        steps = 0
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
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
            try:
                output_list = layout_net.forward(*transform_sample(sample))
            except ProcessingException:
                continue  # skip example
            pred = make_prediction(output_list, device)
            if loss is None:
                print(pred.requires_grad, label.requires_grad)
                loss = loss_func(pred, label)
                if torch.isnan(loss).data:
                    print("Stop training because loss=%s" % (loss.data))
                    stop_training = True
                    break
            else:
                l = loss_func(pred, label)
                if torch.isnan(l).data:
                    print("Stop training because loss=%s" % (l.data))
                    stop_training = True
                    break
                loss += l
            steps += 1
            writer_it += 1  # this way the number in tensorboard will correspond to the actual number of iterations
            accuracy.append(int(torch.sigmoid(pred) == label))
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
                print(f"Stop training because maximum number of steps {steps} has been performed")
                stop_training = True
                break
        writer.add_scalar("Training Loss/train",
                          np.mean(cumulative_loss[-int(print_every / batch_size):]), writer_it)
        writer.add_scalar("Training Acc/train",
                          np.mean(accuracy[-print_every:]), writer_it)
        layout_net.set_eval()
        mrr, p_at_ks = eval_mrr_and_p_at_k(valid_data, layout_net, device, k, distractor_set_size, count=250)
        acc = eval_acc(valid_data, layout_net, count=1000, device=device)
        writer.add_scalar("Training MRR/valid", mrr, writer_it)
        for pre, ki in zip(p_at_ks, k):
            writer.add_scalar(f"Training P@{k}/valid", pre, writer_it)
        writer.add_scalar("Training Acc/valid", acc, writer_it)
        cur_perf = (mrr, acc, p_at_ks[0])
        print("Best performance: ", best_accuracy)
        print("Current performance: ", cur_perf)
        print("best < current: ", best_accuracy < cur_perf)
        if best_accuracy < cur_perf:
            layout_net.save_to_checkpoint(checkpoint_dir + '/best_model.tar')
            print("Saving model with best training performance (mrr, acc, p@k): %s -> %s on epoch=%d, global_step=%d" %
                  (best_accuracy, cur_perf, epoch, steps))
            best_accuracy = cur_perf
            wait_step = 0
            stop_training = False
        else:
            wait_step += 1
            if wait_step >= patience:
                print("Stopping training because wait steps exceeded: ", wait_step)
                stop_training = True
        layout_net.set_train()
        if use_lr_scheduler:
            scheduler.step(np.mean(cumulative_loss[-print_every:]))
        if stop_training:
            break


def main(device, data_dir, scoring_checkpoint, num_epochs, num_epochs_pretraining, lr, print_every,
         valid_file_name, num_negatives, adamw,
         example_count, dropout, checkpoint_dir, summary_writer_dir, use_lr_scheduler,
         clip_grad_value, patience, k, distractor_set_size, do_pretrain, do_train, batch_size, layout_net_training_ckp,
         finetune_scoring):
    shard_range = num_negatives
    dataset = ConcatDataset(
        [CodeSearchNetDataset_wShards(data_dir, r, shard_it, device) for r in range(1) for shard_it in
         range(shard_range + 1)])

    if valid_file_name != "None":
        valid_data = CodeSearchNetDataset_NotPrecomputed(filename=valid_file_name, device=device)
    else:
        valid_data, dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.3), int(len(dataset) * 0.7)],
                                                            generator=torch.Generator().manual_seed(42))
        print("Len of validation dataset before filtering: ", len(valid_data))
        valid_data = filter_neg_samples(valid_data, device)
        print("Len of validation dataset after filtering: ", len(valid_data))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    scoring_module = ScoringModule(device, scoring_checkpoint)
    action_module = ActionModule(device, dim_size=embedder.dim, dropout=dropout)
    layout_net = LayoutNet(scoring_module, action_module, device)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    writer = SummaryWriter(summary_writer_dir + f'/{dt_string}')
    print("Writing to tensorboard: ", dt_string)
    checkpoint_dir = checkpoint_dir + f'/{dt_string}'
    print("Checkpoints will be saved in ", checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if do_pretrain:
        pretrain(layout_net=layout_net, adamw=adamw, checkpoint_dir=checkpoint_dir, num_epochs=num_epochs_pretraining,
                 data_loader=data_loader, clip_grad_value=clip_grad_value, example_count=example_count, device=device,
                 lr=lr, print_every=print_every, writer=writer, k=k, valid_data=valid_data,
                 distractor_set_size=distractor_set_size, patience=patience, use_lr_scheduler=use_lr_scheduler,
                 batch_size=batch_size)
    if do_train:
        if layout_net_training_ckp is not None:
            layout_net.load_from_checkpoint(layout_net_training_ckp)
        train(layout_net=layout_net, device=device, lr=lr, adamw=adamw, checkpoint_dir=checkpoint_dir,
              num_epochs=num_epochs, data_loader=data_loader, clip_grad_value=clip_grad_value,
              example_count=example_count, use_lr_scheduler=use_lr_scheduler,
              writer=writer, valid_data=valid_data, k=k, distractor_set_size=distractor_set_size,
              print_every=print_every, patience=patience, batch_size=batch_size, finetune_scoring=finetune_scoring)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end training of neural module network')
    parser.add_argument('--device', dest='device', type=str, help='device to run on')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='training data directory', required=True)
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str,
                        help='Scoring module checkpoint', required=True)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, help='number of epochs to train', default=50)
    parser.add_argument('--num_epochs_pretraining', dest='num_epochs_pretraining', type=int,
                        help='number of epochs to train', default=50)
    parser.add_argument('--print_every', dest='print_every', type=int,
                        help='print to tensorboard after this many iterations', default=1000)
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate', required=True)
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='Validation data file', required=True)
    parser.add_argument('--num_negatives', dest='num_negatives', type=int,
                        help='Number of distractors to use in training')
    parser.add_argument('--adamw', dest='adamw', type=float, default=0)
    parser.add_argument('--example_count', dest='example_count', type=int, default=0)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str)
    parser.add_argument('--summary_writer_dir', dest='summary_writer_dir', type=str)
    parser.add_argument('--use_lr_scheduler', dest='use_lr_scheduler', default=False, action='store_true')
    parser.add_argument('--clip_grad_value', dest='clip_grad_value', default=0, type=float)
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--p_at_k', dest='p_at_k', action='append')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=20)
    parser.add_argument('--distractor_set_size', dest='distractor_set_size', type=int, default=1000)
    parser.add_argument('--do_pretrain', dest='do_pretrain', default=False, action='store_true')
    parser.add_argument('--do_train', dest='do_train', default=False, action='store_true')
    parser.add_argument('--finetune_scoring', dest='finetune_scoring', default=False, action='store_true')
    parser.add_argument('--layout_net_training_ckp', dest='layout_net_training_ckp', type=str)

    args = parser.parse_args()
    main(device=args.device,
         data_dir=args.data_dir,
         scoring_checkpoint=args.scoring_checkpoint,
         num_epochs=args.num_epochs,
         num_epochs_pretraining=args.num_epochs_pretraining,
         lr=args.lr,
         print_every=args.print_every,
         valid_file_name=args.valid_file_name,
         num_negatives=args.num_negatives,
         adamw=args.adamw,
         example_count=args.example_count,
         dropout=args.dropout,
         checkpoint_dir=args.checkpoint_dir,
         summary_writer_dir=args.summary_writer_dir,
         use_lr_scheduler=args.use_lr_scheduler,
         clip_grad_value=args.clip_grad_value,
         patience=args.patience,
         k=[int(ki) for ki in args.p_at_k],
         distractor_set_size=args.distractor_set_size,
         do_pretrain=args.do_pretrain,
         do_train=args.do_train,
         batch_size=args.batch_size,
         layout_net_training_ckp=args.layout_net_training_ckp,
         finetune_scoring=args.finetune_scoring)
