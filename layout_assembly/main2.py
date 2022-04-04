import argparse
import os
from datetime import datetime

import glob
import numpy as np
import torch
import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import codebert_embedder_v2 as embedder
from action2.action import ActionModule
from eval.dataset import CodeSearchNetDataset_NotPrecomputed, CodeSearchNetDataset_NotPrecomputed_RandomNeg, \
    CodeSearchNetDataset_NegativeOracleNotPrecomputed, CodeSearchNetDataset_InBatchNegativesOracle
from eval.dataset import transform_sample, filter_neg_samples
from layout_assembly.layout_ws2 import LayoutNetWS2 as LayoutNet
from layout_assembly.modules import ScoringModule
from layout_assembly.utils import ProcessingException
from utils import binarize, eval_mrr_and_p_at_k, eval_acc, eval_acc_f1_pretraining_task, get_alignment_function


class LearningRateWarmUP(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.cur_iteration = 0
        self.step(1)

    def warmup_learning_rate(self):
        warmup_lr = self.target_lr * float(self.cur_iteration) / float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, perf):
        if self.cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate()
        else:
            self.after_scheduler.step(perf)
        self.cur_iteration += 1

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)


def pretrain(layout_net, lr, adamw, checkpoint_dir, num_epochs, data_loader, clip_grad_value, device, print_every,
             writer, k, valid_data, distractor_set_size, patience, use_lr_scheduler, batch_size, skip_negatives,
             override_negatives, threshold, loss_type):
    if loss_type == 'bce_loss':
        loss_func = torch.nn.BCELoss()
    elif loss_type == 'kldiv_loss':
        loss_func = torch.nn.KLDivLoss(reduction='batchmean')
    elif loss_type == 'mse_loss':
        loss_func = torch.nn.MSELoss()
    op = torch.optim.Adam(layout_net.parameters(), lr=lr, weight_decay=adamw)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, verbose=True)
    layout_net.set_train()
    checkpoint_dir += '/pretrain'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    total_steps = 0
    best_accuracy = (-1.0, -1.0)
    wait_step = 0
    stop_training = False

    for epoch in range(num_epochs):
        layout_net.set_train()
        if stop_training:
            break
        cumulative_loss = []
        accuracy = []
        f1_scores = []
        loss = None
        epoch_steps = 0
        for x in layout_net.parameters():
            x.grad = None
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            if stop_training:
                break
            sample, _, _, label = datum
            if skip_negatives:
                if label == 0:  # skip negative samples
                    continue
            try:
                output_list = layout_net.forward(*transform_sample(sample))
            except ProcessingException:
                continue
            for out in output_list:
                true_out, pred_out = out
                if override_negatives:
                    if label == 0:
                        true_out = torch.ones_like(true_out) * 1e-8

                labels = binarize(true_out, threshold=threshold).to(device)
                if loss_type == 'bce_loss':
                    l = loss_func(pred_out, labels)
                elif loss_type == 'kldiv_loss':
                    norm_true_out = true_out/(torch.sum(true_out)) # normalize to probability
                    norm_pred_out = torch.logit(pred_out / (torch.sum(pred_out)))
                    l = loss_func(norm_pred_out, norm_true_out)
                elif loss_type == 'mse_loss':
                    l = loss_func(pred_out, true_out)
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
                binarized_preds = binarize(pred_out, threshold=threshold)
                accuracy.append(sum((binarized_preds == labels).cpu().detach().numpy()) * 1. / labels.shape[0])
                f1_scores.append(f1_score(labels.cpu().detach().numpy().flatten(),
                                          binarized_preds.cpu().detach().numpy().flatten(), zero_division=1))

            epoch_steps += 1
            total_steps += 1  # this way the number in tensorboard will correspond to the actual number of iterations
            if epoch_steps % batch_size == 0:
                loss.backward()
                cumulative_loss.append(loss.data.cpu().numpy() / batch_size)
                if clip_grad_value > 0:
                    torch.nn.utils.clip_grad_value_(layout_net.parameters(), clip_grad_value)
                op.step()
                loss = None
                for x in layout_net.parameters():
                    x.grad = None
            if epoch_steps % print_every == 0:
                writer.add_scalar("Pretraining Loss/pretraining",
                                  np.mean(cumulative_loss[-int(print_every / batch_size):]), total_steps)
                writer.add_scalar("Pretraining Acc/train pretraining",
                                  np.mean(accuracy[-print_every:]), total_steps)
                writer.add_scalar("Pretraining F1/pretraining",
                                  np.mean(f1_scores[-print_every:]), total_steps)
                layout_net.set_eval()
                acc, f1 = eval_acc_f1_pretraining_task(valid_data, layout_net, override_negatives=override_negatives,
                                                       threshold=threshold, count=1000)
                writer.add_scalar("Pretraining F1/valid pretraining", f1, total_steps)
                writer.add_scalar("Pretraining Acc/valid pretraining", acc, total_steps)
                cur_perf = (f1, acc)
                print("Current pretraining performance: ", cur_perf, ", best pretraining performance: ", best_accuracy)
                if best_accuracy < cur_perf:
                    layout_net.save_to_checkpoint(checkpoint_dir + '/best_model.tar')
                    print(
                        "Saving model with best pretraining accuracy performance: %s -> %s on epoch=%d, global_step=%d" %
                        (best_accuracy, cur_perf, epoch, epoch_steps))
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


def train(device, layout_net, lr, adamw, checkpoint_dir, num_epochs, data_loader, clip_grad_value, use_lr_scheduler,
          writer, valid_data, k, distractor_set_size, print_every, patience, batch_size,
          make_prediction, use_warmup_lr, warmup_steps, use_in_batch_negatives, use_margin_loss, optim_type='adam'):
    if use_in_batch_negatives:
        train_inbatch_neg(device=device, layout_net=layout_net, lr=lr, adamw=adamw, checkpoint_dir=checkpoint_dir,
              num_epochs=num_epochs, data_loader=data_loader, clip_grad_value=clip_grad_value,
              use_lr_scheduler=use_lr_scheduler, writer=writer, valid_data=valid_data, k=k,
              distractor_set_size=distractor_set_size, print_every=print_every, patience=patience,
              batch_size=batch_size, make_prediction=make_prediction, use_warmup_lr=use_warmup_lr,
              warmup_steps=warmup_steps)
    elif use_margin_loss:
        train_margin_ranking(device=device, layout_net=layout_net, lr=lr, adamw=adamw, checkpoint_dir=checkpoint_dir,
              num_epochs=num_epochs, data_loader=data_loader, clip_grad_value=clip_grad_value,
              use_lr_scheduler=use_lr_scheduler, writer=writer, valid_data=valid_data, k=k,
              distractor_set_size=distractor_set_size, print_every=print_every, patience=patience,
              batch_size=batch_size, make_prediction=make_prediction, use_warmup_lr=use_warmup_lr,
              warmup_steps=warmup_steps)
    loss_func = torch.nn.BCELoss()
    if optim_type == 'sgd':
        op = torch.optim.SGD(layout_net.parameters(), lr=lr, weight_decay=adamw)
    elif optim_type == 'adam':
        op = torch.optim.Adam(layout_net.parameters(), lr=lr, weight_decay=adamw)
    else:
        raise Exception("Unknown optimizer type!! ", optim_type)
    if layout_net.weighted_cosine:  # try a hack?
        op.add_param_group({"params": layout_net.weight})
    if layout_net.weighted_cosine_v2:  # try a hack?
        op.add_param_group({"params": layout_net.weight.parameters()})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, verbose=True, patience=1000, factor=0.5, min_lr=1e-8)
    if use_warmup_lr:
        scheduler = LearningRateWarmUP(optimizer=op, warmup_iteration=warmup_steps, target_lr=lr,
                                       after_scheduler=scheduler)

    checkpoint_dir += '/train'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    positive_label = torch.tensor(1, dtype=torch.float).to(device)
    negative_label = torch.tensor(0, dtype=torch.float).to(device)
    total_steps = 0
    best_accuracy = (-1.0, -1.0, -1.0)
    wait_step = 0
    stop_training = False

    for epoch in range(num_epochs):
        layout_net.set_train()
        if stop_training:
            break
        cumulative_loss = []
        accuracy = []
        loss = None
        epoch_steps = 0
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            for param in layout_net.parameters():
                param.grad = None
            if layout_net.weighted_cosine:
                layout_net.weight.grad = None
            sample, scores, verbs, label = datum
            if int(label) == 0:
                label = negative_label
            else:
                label = positive_label
            try:
                output_list = layout_net.forward(*transform_sample(sample))
            except ProcessingException:
                continue  # skip example
            pred = make_prediction(output_list)
            if loss is None:
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
            epoch_steps += 1
            total_steps += 1  # this way the number in tensorboard will correspond to the actual number of iterations
            binarized_pred = binarize(pred, threshold=0.5)

            accuracy.append(int((binarized_pred == label).cpu().detach().numpy()))
            if epoch_steps % batch_size == 0:
                loss.backward()
                cumulative_loss.append(loss.data.cpu().numpy() / batch_size)
                if clip_grad_value > 0:
                    torch.nn.utils.clip_grad_value_(layout_net.parameters(), clip_grad_value)
                    if layout_net.weighted_cosine:
                        torch.nn.utils.clip_grad_value_(layout_net.weight, clip_grad_value)
                    if layout_net.weighted_cosine_v2:
                        torch.nn.utils.clip_grad_value_(layout_net.weight.parameters(), clip_grad_value)
                op.step()
                if use_warmup_lr:
                    scheduler.step(np.mean(cumulative_loss[-print_every:]))
                loss = None
                for x in layout_net.parameters():
                    x.grad = None
                if layout_net.weighted_cosine:
                    layout_net.weight.grad = None
                if layout_net.weighted_cosine_v2:
                    for x in layout_net.weight.parameters():
                        x.grad = None
            if epoch_steps % print_every == 0:
                writer.add_scalar("Training Loss/train",
                                  np.mean(cumulative_loss[-int(print_every / batch_size):]), total_steps)
                writer.add_scalar("Training Acc/train",
                                  np.mean(accuracy[-print_every:]), total_steps)
                layout_net.set_eval()
                mrr, p_at_ks = eval_mrr_and_p_at_k(dataset=valid_data, layout_net=layout_net, k=k,
                                                   distractor_set_size=distractor_set_size,
                                                   make_prediction=make_prediction, count=20)
                acc = eval_acc(dataset=valid_data, layout_net=layout_net, make_prediction=make_prediction, count=500)
                writer.add_scalar("Training MRR/valid", mrr, total_steps)
                for pre, ki in zip(p_at_ks, k):
                    writer.add_scalar(f"Training P@{ki}/valid", pre, total_steps)
                writer.add_scalar("Training Acc/valid", acc, total_steps)
                cur_perf = (mrr, acc, p_at_ks[0])
                print("Current performance: ", cur_perf, ", best performance: ", best_accuracy)
                if best_accuracy < cur_perf:
                    layout_net.save_to_checkpoint(checkpoint_dir + '/best_model.tar')
                    print(
                        "Saving model with best training performance (mrr, acc, p@k): %s -> %s on epoch=%d, global_step=%d" %
                        (best_accuracy, cur_perf, epoch, epoch_steps))
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

        # end of epoch eval
        writer.add_scalar("Training Loss/train",
                          np.mean(cumulative_loss[-int(print_every / batch_size):]), total_steps)
        writer.add_scalar("Training Acc/train",
                          np.mean(accuracy[-print_every:]), total_steps)
        layout_net.set_eval()
        mrr, p_at_ks = eval_mrr_and_p_at_k(dataset=valid_data, layout_net=layout_net, k=k,
                                           distractor_set_size=distractor_set_size, make_prediction=make_prediction,
                                           count=20)
        acc = eval_acc(dataset=valid_data, layout_net=layout_net, make_prediction=make_prediction, count=500)

        writer.add_scalar("Training MRR/valid", mrr, total_steps)
        for pre, ki in zip(p_at_ks, k):
            writer.add_scalar(f"Training P@{k}/valid", pre, total_steps)
        writer.add_scalar("Training Acc/valid", acc, total_steps)


def train_inbatch_neg(device, layout_net, lr, adamw, checkpoint_dir, num_epochs, data_loader, clip_grad_value, use_lr_scheduler,
          writer, valid_data, k, distractor_set_size, print_every, patience, batch_size,
          make_prediction, use_warmup_lr, warmup_steps, optim_type='adam'):
    loss_func = torch.nn.BCELoss()
    if optim_type == 'sgd':
        op = torch.optim.SGD(layout_net.parameters(), lr=lr, weight_decay=adamw)
    elif optim_type == 'adam':
        op = torch.optim.Adam(layout_net.parameters(), lr=lr, weight_decay=adamw)
    else:
        raise Exception("Unknown optimizer type!! ", optim_type)
    if layout_net.weighted_cosine:  # try a hack?
        op.add_param_group({"params": layout_net.weight})
    if layout_net.weighted_cosine_v2:  # try a hack?
        op.add_param_group({"params": layout_net.weight.parameters()})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, verbose=True, patience=1000, factor=0.5, min_lr=1e-8)
    if use_warmup_lr:
        scheduler = LearningRateWarmUP(optimizer=op, warmup_iteration=warmup_steps, target_lr=lr,
                                       after_scheduler=scheduler)

    checkpoint_dir += '/train'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    positive_label = torch.tensor(1, dtype=torch.float).to(device)
    negative_label = torch.tensor(0, dtype=torch.float).to(device)
    total_steps = 0
    best_accuracy = (-1.0, -1.0, -1.0)
    wait_step = 0
    stop_training = False

    for epoch in range(num_epochs):
        layout_net.set_train()
        if stop_training:
            break
        cumulative_loss = []
        accuracy = []
        loss = None
        epoch_steps = 0
        for i, batch in tqdm.tqdm(enumerate(data_loader)):
            batch_size = 0
            for datum in batch:
                for param in layout_net.parameters():
                    param.grad = None
                if layout_net.weighted_cosine:
                    layout_net.weight.grad = None
                sample, scores, verbs, label = datum
                if int(label) == 0:
                    label = negative_label
                else:
                    label = positive_label
                try:
                    output_list = layout_net.forward(*transform_sample(sample))
                except ProcessingException:
                    continue  # skip example
                pred = make_prediction(output_list)
                if loss is None:
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
                epoch_steps += 1
                batch_size += 1
                total_steps += 1  # this way the number in tensorboard will correspond to the actual number of iterations
                binarized_pred = binarize(pred, threshold=0.5)

                accuracy.append(int((binarized_pred == label).cpu().detach().numpy()))
            if loss:
                loss.backward()
                cumulative_loss.append(loss.data.cpu().numpy() / batch_size)
                if clip_grad_value > 0:
                    torch.nn.utils.clip_grad_value_(layout_net.parameters(), clip_grad_value)
                    if layout_net.weighted_cosine:
                        torch.nn.utils.clip_grad_value_(layout_net.weight, clip_grad_value)
                    if layout_net.weighted_cosine_v2:
                        torch.nn.utils.clip_grad_value_(layout_net.weight.parameters(), clip_grad_value)
                op.step()
                if use_warmup_lr:
                    scheduler.step(np.mean(cumulative_loss[-print_every:]))
            loss = None
            for x in layout_net.parameters():
                x.grad = None
            if layout_net.weighted_cosine:
                layout_net.weight.grad = None
            if layout_net.weighted_cosine_v2:
                for x in layout_net.weight.parameters():
                    x.grad = None
            if epoch_steps % print_every <= batch_size:
                writer.add_scalar("Training Loss/train",
                                  np.mean(cumulative_loss[-int(print_every/batch_size):]), total_steps)
                writer.add_scalar("Training Acc/train",
                                  np.mean(accuracy[-print_every:]), total_steps)
                layout_net.set_eval()
                mrr, p_at_ks = eval_mrr_and_p_at_k(dataset=valid_data, layout_net=layout_net, k=k,
                                                   distractor_set_size=distractor_set_size,
                                                   make_prediction=make_prediction, count=20)
                acc = eval_acc(dataset=valid_data, layout_net=layout_net, make_prediction=make_prediction, count=500)
                writer.add_scalar("Training MRR/valid", mrr, total_steps)
                for pre, ki in zip(p_at_ks, k):
                    writer.add_scalar(f"Training P@{ki}/valid", pre, total_steps)
                writer.add_scalar("Training Acc/valid", acc, total_steps)
                cur_perf = (mrr, acc, p_at_ks[0])
                print("Current performance: ", cur_perf, ", best performance: ", best_accuracy)
                if best_accuracy < cur_perf:
                    layout_net.save_to_checkpoint(checkpoint_dir + '/best_model.tar')
                    print(
                        "Saving model with best training performance (mrr, acc, p@k): %s -> %s on epoch=%d, global_step=%d" %
                        (best_accuracy, cur_perf, epoch, epoch_steps))
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

        # end of epoch eval
        writer.add_scalar("Training Loss/train",
                          np.mean(cumulative_loss[-int(print_every / batch_size):]), total_steps)
        writer.add_scalar("Training Acc/train",
                          np.mean(accuracy[-print_every:]), total_steps)
        layout_net.set_eval()
        mrr, p_at_ks = eval_mrr_and_p_at_k(dataset=valid_data, layout_net=layout_net, k=k,
                                           distractor_set_size=distractor_set_size, make_prediction=make_prediction,
                                           count=20)
        acc = eval_acc(dataset=valid_data, layout_net=layout_net, make_prediction=make_prediction, count=500)

        writer.add_scalar("Training MRR/valid", mrr, total_steps)
        for pre, ki in zip(p_at_ks, k):
            writer.add_scalar(f"Training P@{k}/valid", pre, total_steps)
        writer.add_scalar("Training Acc/valid", acc, total_steps)


def train_margin_ranking(device, layout_net, lr, adamw, checkpoint_dir, num_epochs, data_loader, clip_grad_value, use_lr_scheduler,
          writer, valid_data, k, distractor_set_size, print_every, patience, batch_size,
          make_prediction, use_warmup_lr, warmup_steps, optim_type='adam'):
    loss_func = torch.nn.MarginRankingLoss()
    if optim_type == 'sgd':
        op = torch.optim.SGD(layout_net.parameters(), lr=lr, weight_decay=adamw)
    elif optim_type == 'adam':
        op = torch.optim.Adam(layout_net.parameters(), lr=lr, weight_decay=adamw)
    else:
        raise Exception("Unknown optimizer type!! ", optim_type)
    if layout_net.weighted_cosine:  # try a hack?
        op.add_param_group({"params": layout_net.weight})
    if layout_net.weighted_cosine_v2:  # try a hack?
        op.add_param_group({"params": layout_net.weight.parameters()})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, verbose=True, patience=1000, factor=0.5, min_lr=1e-8)
    if use_warmup_lr:
        scheduler = LearningRateWarmUP(optimizer=op, warmup_iteration=warmup_steps, target_lr=lr,
                                       after_scheduler=scheduler)

    checkpoint_dir += '/train'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    total_steps = 0
    best_accuracy = (-1.0, -1.0, -1.0)
    wait_step = 0
    stop_training = False

    for epoch in range(num_epochs):
        layout_net.set_train()
        if stop_training:
            break
        cumulative_loss = []
        accuracy = []
        loss = None
        epoch_steps = 0
        for i, batch in tqdm.tqdm(enumerate(data_loader)):
            batch_size = 0
            zero_grads(layout_net)
            sample, scores, verbs, label = batch[0]
            assert int(label) == 1
            try:
                pos_out = layout_net.forward(*transform_sample(sample))
                pos_pred = make_prediction(pos_out)
            except ProcessingException:
                continue
            binarized_pred = binarize(pos_pred, threshold=0.5)
            accuracy.append(int((binarized_pred == 1).cpu().detach().numpy()))
            for datum in batch[1:]:
                zero_grads(layout_net)
                sample, scores, verbs, label = datum
                assert int(label) == 0
                try:
                    neg_out = layout_net.forward(*transform_sample(sample))
                except ProcessingException:
                    continue  # skip example
                neg_pred = make_prediction(neg_out)
                if np.random.rand() > 0.5:
                    loss = loss_func(pos_pred, neg_pred, 1)
                else:
                    loss = loss_func(neg_pred, pos_pred, -1)
                if torch.isnan(loss).data:
                    print("Stop training because loss=%s" % (loss.data))
                    stop_training = True
                    break
                epoch_steps += 1
                batch_size += 1
                total_steps += 1  # this way the number in tensorboard will correspond to the actual number of iterations
                binarized_pred = binarize(neg_pred, threshold=0.5)
                accuracy.append(int((binarized_pred == 0).cpu().detach().numpy()))
                loss.backward()
                cumulative_loss.append(loss.data.cpu().numpy() / batch_size)
                if clip_grad_value > 0:
                    torch.nn.utils.clip_grad_value_(layout_net.parameters(), clip_grad_value)
                    if layout_net.weighted_cosine:
                        torch.nn.utils.clip_grad_value_(layout_net.weight, clip_grad_value)
                    if layout_net.weighted_cosine_v2:
                        torch.nn.utils.clip_grad_value_(layout_net.weight.parameters(), clip_grad_value)
                op.step()
                if use_warmup_lr:
                    scheduler.step(np.mean(cumulative_loss[-print_every:]))
            zero_grads(layout_net)
            if epoch_steps % print_every <= batch_size:
                writer.add_scalar("Training Loss/train",
                                  np.mean(cumulative_loss[-int(print_every/batch_size):]), total_steps)
                writer.add_scalar("Training Acc/train",
                                  np.mean(accuracy[-print_every:]), total_steps)
                layout_net.set_eval()
                mrr, p_at_ks = eval_mrr_and_p_at_k(dataset=valid_data, layout_net=layout_net, k=k,
                                                   distractor_set_size=distractor_set_size,
                                                   make_prediction=make_prediction, count=20)
                acc = eval_acc(dataset=valid_data, layout_net=layout_net, make_prediction=make_prediction, count=500)
                writer.add_scalar("Training MRR/valid", mrr, total_steps)
                for pre, ki in zip(p_at_ks, k):
                    writer.add_scalar(f"Training P@{ki}/valid", pre, total_steps)
                writer.add_scalar("Training Acc/valid", acc, total_steps)
                cur_perf = (mrr, acc, p_at_ks[0])
                print("Current performance: ", cur_perf, ", best performance: ", best_accuracy)
                if best_accuracy < cur_perf:
                    layout_net.save_to_checkpoint(checkpoint_dir + '/best_model.tar')
                    print(
                        "Saving model with best training performance (mrr, acc, p@k): %s -> %s on epoch=%d, global_step=%d" %
                        (best_accuracy, cur_perf, epoch, epoch_steps))
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

        # end of epoch eval
        writer.add_scalar("Training Loss/train",
                          np.mean(cumulative_loss[-int(print_every / batch_size):]), total_steps)
        writer.add_scalar("Training Acc/train",
                          np.mean(accuracy[-print_every:]), total_steps)
        layout_net.set_eval()
        mrr, p_at_ks = eval_mrr_and_p_at_k(dataset=valid_data, layout_net=layout_net, k=k,
                                           distractor_set_size=distractor_set_size, make_prediction=make_prediction,
                                           count=20)
        acc = eval_acc(dataset=valid_data, layout_net=layout_net, make_prediction=make_prediction, count=500)

        writer.add_scalar("Training MRR/valid", mrr, total_steps)
        for pre, ki in zip(p_at_ks, k):
            writer.add_scalar(f"Training P@{k}/valid", pre, total_steps)
        writer.add_scalar("Training Acc/valid", acc, total_steps)


def zero_grads(layout_net):
    for param in layout_net.parameters():
        param.grad = None
    if layout_net.weighted_cosine:
        layout_net.weight.grad = None
    if layout_net.weighted_cosine_v2:
        for x in layout_net.weight.parameters():
            x.grad = None


def eval(layout_net, data, k, distractor_set_size, make_prediction, count):
    layout_net.set_eval()
    mrr, p_at_ks = eval_mrr_and_p_at_k(dataset=data, layout_net=layout_net, make_prediction=make_prediction,
                                       k=k, distractor_set_size=distractor_set_size, count=count)
    acc = eval_acc(dataset=data, layout_net=layout_net, count=count, make_prediction=make_prediction)

    print("MRR: ", mrr)
    for pre, ki in zip(p_at_ks, k):
        print(f"P@{k}", pre)
    print("Acc", acc)
    layout_net.set_train()


def main(device, data_dir, scoring_checkpoint, num_epochs, num_epochs_pretraining, lr, print_every, valid_file_name,
         num_negatives, adamw, example_count, dropout, checkpoint_dir, summary_writer_dir, use_lr_scheduler,
         clip_grad_value, patience, k, distractor_set_size, do_pretrain, do_train, batch_size, layout_net_training_ckp,
         finetune_scoring, override_negatives_in_pretraining, skip_negatives_in_pretraining, use_dummy_action, do_eval,
         alignment_function, pretrain_bin_threshold, pretrain_loss_type, eval_count, use_warmup_lr, warmup_steps,
         oracle_idxs_file, use_in_batch_negatives, use_margin_loss):
    if os.path.isfile(data_dir):
        print(f"Loading dataset from {data_dir}")
        if oracle_idxs_file:
            if use_in_batch_negatives:
                dataset = CodeSearchNetDataset_InBatchNegativesOracle(filename=data_dir, device=device,
                                                                      neg_count=num_negatives, oracle_idxs_file=oracle_idxs_file)
            else:
                dataset = ConcatDataset([CodeSearchNetDataset_NotPrecomputed(data_dir, device), ] +
                                    [CodeSearchNetDataset_NegativeOracleNotPrecomputed(filename=data_dir, device=device,
                                                                                       neg_count=num_negatives,
                                                                                       oracle_idxs_file=oracle_idxs_file)])
        else:
            dataset = ConcatDataset([CodeSearchNetDataset_NotPrecomputed(data_dir, device), ] +
                                    [CodeSearchNetDataset_NotPrecomputed_RandomNeg(filename=data_dir, device=device,
                                                                                   range=r) for r in range(1)])
    elif os.path.isdir(data_dir):
        print(f"Loading all files from dataset from {data_dir}")
        d_list = []
        for file in glob.glob(os.path.join(data_dir, "*")):
            print(f"Loading dataset from {file}")
            d_list.extend([CodeSearchNetDataset_NotPrecomputed(file, device), ] +
                                    [CodeSearchNetDataset_NotPrecomputed_RandomNeg(filename=file, device=device,
                                                                                   range=r) for r in range(1)])
        dataset = ConcatDataset(d_list)
    print("Dataset read, the length of the dataset is: ", len(dataset))
    if valid_file_name != "None":
        valid_data = CodeSearchNetDataset_NotPrecomputed(filename=valid_file_name, device=device)
    else:
        valid_data, dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.3), int(len(dataset) * 0.7)],
                                                            generator=torch.Generator().manual_seed(42))
        print("Len of validation dataset before filtering: ", len(valid_data))
        valid_data = filter_neg_samples(valid_data, device)
        print("Len of validation dataset after filtering: ", len(valid_data))
    if example_count > 0 and example_count < len(dataset):
        import torch.utils.data as data_utils
        perm = torch.randperm(len(dataset))
        indices = perm[:example_count]
        dataset = data_utils.Subset(dataset, indices)
        print(f"Modified dataset, new dataset has {len(dataset)} examples")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    scoring_module = ScoringModule(device, scoring_checkpoint)
    action_module = ActionModule(device, dim_size=embedder.dim, dropout=dropout)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    writer = SummaryWriter(summary_writer_dir + f'/{dt_string}')
    print("Writing to tensorboard: ", dt_string)
    checkpoint_dir = checkpoint_dir + f'/{dt_string}'
    print("Checkpoints will be saved in ", checkpoint_dir)

    code_in_output, weighted_cosine, weighted_cosine_v2, mlp_prediction, make_prediction = \
        get_alignment_function(alignment_function)
    layout_net = LayoutNet(scoring_module, action_module, device, code_in_output, weighted_cosine, weighted_cosine_v2,
                           mlp_prediction)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if do_pretrain:
        cio = code_in_output
        wc = weighted_cosine
        wc_2 = weighted_cosine_v2
        mp = mlp_prediction
        layout_net.weighted_cosine = False
        layout_net.weighted_cosine_v2 = False
        layout_net.code_in_output = False
        layout_net.mlp_prediction = False
        pretrain(layout_net=layout_net, lr=lr, adamw=adamw, checkpoint_dir=checkpoint_dir,
                 num_epochs=num_epochs_pretraining, data_loader=data_loader, clip_grad_value=clip_grad_value,
                 device=device, print_every=print_every, writer=writer, k=k, valid_data=valid_data,
                 distractor_set_size=distractor_set_size, patience=patience, use_lr_scheduler=use_lr_scheduler,
                 batch_size=batch_size, skip_negatives=skip_negatives_in_pretraining,
                 override_negatives=override_negatives_in_pretraining, threshold=pretrain_bin_threshold,
                 loss_type=pretrain_loss_type)
        layout_net.weighted_cosine = wc
        layout_net.weighted_cosine_v2 = wc_2
        layout_net.code_in_output = cio
        layout_net.mlp_prediction = mp
    if finetune_scoring:
        layout_net.finetune_scoring = finetune_scoring
    if layout_net_training_ckp is not None:
        layout_net.load_from_checkpoint(layout_net_training_ckp)
    else:
        pretraining_best_checkpoint = checkpoint_dir + '/pretrain/best_model.tar'
        if os.path.exists(pretraining_best_checkpoint):
            layout_net.load_from_checkpoint(pretraining_best_checkpoint)
    if do_train:
        if use_dummy_action:
            from action2.dummy_action import DummyActionModule
            action_module = DummyActionModule(device, dim_size=embedder.dim, dropout=dropout)
            layout_net = LayoutNet(scoring_module, action_module, device)
        train(device=device, layout_net=layout_net, lr=lr, adamw=adamw, checkpoint_dir=checkpoint_dir,
              num_epochs=num_epochs, data_loader=data_loader, clip_grad_value=clip_grad_value,
              use_lr_scheduler=use_lr_scheduler, writer=writer, valid_data=valid_data, k=k,
              distractor_set_size=distractor_set_size, print_every=print_every, patience=patience,
              batch_size=batch_size, make_prediction=make_prediction, use_warmup_lr=use_warmup_lr,
              warmup_steps=warmup_steps, use_in_batch_negatives=use_in_batch_negatives, use_margin_loss=use_margin_loss)
    if do_eval:
        eval(layout_net=layout_net, data=valid_data, k=k, distractor_set_size=distractor_set_size,
             count=eval_count, make_prediction=make_prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end training of neural module network')
    parser.add_argument('--device', dest='device', type=str, help='device to run on')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='training data directory', required=True)
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str,
                        help='Scoring module checkpoint')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, help='number of epochs to train', default=50)
    parser.add_argument('--num_epochs_pretraining', dest='num_epochs_pretraining', type=int,
                        help='number of epochs to train', default=50)
    parser.add_argument('--print_every', dest='print_every', type=int,
                        help='print to tensorboard after this many iterations', default=1000)
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate', required=True)
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='Validation data file')
    parser.add_argument('--num_negatives', dest='num_negatives', type=int,
                        help='Number of distractors to use in training')
    parser.add_argument('--adamw', dest='adamw', type=float, default=0)
    parser.add_argument('--example_count', dest='example_count', type=int, default=0)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str)
    parser.add_argument('--summary_writer_dir', dest='summary_writer_dir', type=str)
    parser.add_argument('--use_lr_scheduler', dest='use_lr_scheduler', default=False, action='store_true')
    parser.add_argument('--clip_grad_value', dest='clip_grad_value', default=0, type=float)
    parser.add_argument('--patience', dest='patience', type=int, default=1000)
    parser.add_argument('--p_at_k', dest='p_at_k', action='append')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=20)
    parser.add_argument('--distractor_set_size', dest='distractor_set_size', type=int, default=999)
    parser.add_argument('--do_pretrain', dest='do_pretrain', default=False, action='store_true')
    parser.add_argument('--do_train', dest='do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', dest='do_eval', default=False, action='store_true')
    parser.add_argument('--finetune_scoring', dest='finetune_scoring', default=False, action='store_true')
    parser.add_argument('--layout_net_training_ckp', dest='layout_net_training_ckp', type=str)
    parser.add_argument('--override_negatives_in_pretraining', dest='override_negatives_in_pretraining', default=False,
                        action='store_true')
    parser.add_argument('--skip_negatives_in_pretraining', dest='skip_negatives_in_pretraining', default=False,
                        action='store_true')
    parser.add_argument('--use_dummy_action', dest='use_dummy_action', default=False, action='store_true')
    parser.add_argument('--alignment_function', dest='alignment_function', type=str)
    parser.add_argument('--pretrain_bin_threshold', dest='pretrain_bin_threshold', type=float)
    parser.add_argument('--pretrain_loss_type', dest='pretrain_loss_type', type=str)
    parser.add_argument('--use_warmup_lr', dest='use_warmup_lr', default=False, action='store_true')
    parser.add_argument('--eval_count', dest='eval_count', type=int, default=100,
                        help='How many examples to use in evaluation, pass -1 for evaluating on the entire validation set')
    parser.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=500)
    parser.add_argument('--oracle_idxs_file', dest='oracle_idxs_file', default="", type=str)
    parser.add_argument('--use_in_batch_negatives', dest='use_in_batch_negatives', default=False, action='store_true')
    parser.add_argument('--use_margin_loss', dest='use_margin_loss', default=False, action='store_true')

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
         do_eval=args.do_eval,
         batch_size=args.batch_size,
         layout_net_training_ckp=args.layout_net_training_ckp,
         finetune_scoring=args.finetune_scoring,
         override_negatives_in_pretraining=args.override_negatives_in_pretraining,
         skip_negatives_in_pretraining=args.skip_negatives_in_pretraining,
         use_dummy_action=args.use_dummy_action,
         alignment_function=args.alignment_function,
         pretrain_bin_threshold=args.pretrain_bin_threshold,
         pretrain_loss_type=args.pretrain_loss_type,
         eval_count=args.eval_count,
         use_warmup_lr=args.use_warmup_lr,
         warmup_steps=args.warmup_steps,
         oracle_idxs_file=args.oracle_idxs_file,
         use_in_batch_negatives=args.use_in_batch_negatives,
         use_margin_loss=args.use_margin_loss)
