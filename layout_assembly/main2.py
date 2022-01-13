import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import codebert_embedder_v2 as embedder
from action2.action import ActionModule
from eval.dataset import CodeSearchNetDataset_NotPrecomputed, CodeSearchNetDataset_NotPrecomputed_RandomNeg
from eval.dataset import transform_sample, filter_neg_samples
from eval.utils import mrr, p_at_k
from layout_assembly.layout_ws2 import LayoutNetWS2 as LayoutNet
from layout_assembly.modules import ScoringModule
from layout_assembly.utils import ProcessingException


def create_neg_sample(orig, distr):
    return (orig[0], distr[1], distr[2], distr[3], orig[4])


def binarize(a, threshold):
    with torch.no_grad():
        return torch.where(a < threshold, torch.zeros_like(a), torch.ones_like(a))


def compute_alignment(a, b):
    return torch.dot(a, b)


def make_prediction_weighted_embedding(output_list):
    alignment_scores = None
    cos = torch.nn.CosineSimilarity(dim=0)
    for i in range(len(output_list)):
        a, b, code = output_list[i]
        N = min(a.shape[0], b.shape[0], code.shape[0])
        weighted_code_a = torch.mm(a[:N, :].T, code[:N, :]).squeeze()
        weighted_code_b = torch.mm(b[:N, :].T, code[:N, :]).squeeze()
        s = cos(weighted_code_a, weighted_code_b)
        s = (s + 1) * 0.5
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_weighted_cosine(output_list):
    alignment_scores = None
    cos = torch.nn.CosineSimilarity(dim=0)
    output_list, v = output_list
    for i in range(len(output_list)):
        a, b, code = output_list[i]
        N = min(a.shape[0], b.shape[0], code.shape[0])
        weighted_code_a = torch.mm(a[:N, :].T, code[:N, :]).squeeze()
        weighted_a = v.squeeze() * weighted_code_a
        weighted_code_b = torch.mm(b[:N, :].T, code[:N, :]).squeeze()
        weighted_b = v.squeeze() * weighted_code_b
        s = cos(weighted_a, weighted_b)
        s = (s + 1) * 0.5
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_weighted_cosine_v2(output_list):
    alignment_scores = None
    cos = torch.nn.CosineSimilarity(dim=0)
    output_list, v = output_list
    for i in range(len(output_list)):
        a, b, code = output_list[i]
        N = min(a.shape[0], b.shape[0], code.shape[0])
        a_norm = a/torch.sum(a[:N, :])
        b_norm = b/torch.sum(b[:N, :])
        weighted_code_a = torch.mm(a_norm[:N, :].T, code[:N, :]).squeeze()
        weighted_a = v.squeeze() * weighted_code_a
        weighted_code_b = torch.mm(b_norm[:N, :].T, code[:N, :]).squeeze()
        weighted_b = v.squeeze() * weighted_code_b
        s = cos(weighted_a, weighted_b)
        s = (s + 1) * 0.5
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_dot(output_list):
    alignment_scores = None
    for i in range(len(output_list)):
        s = torch.dot(output_list[i][0].squeeze(), output_list[i][1].squeeze())
        s /= sum(output_list[i][0]).squeeze()
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_cosine(output_list):
    cos = torch.nn.CosineSimilarity(dim=0)
    alignment_scores = None
    for i in range(len(output_list)):
        s = cos(output_list[i][0].squeeze(), output_list[i][1].squeeze())
        s = (s + 1) * 0.5
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def eval_mrr_and_p_at_k(dataset, layout_net, make_prediction, k=[1], distractor_set_size=100, count=250):
    def get_mrr_for_one_sample(dataset, idx, idxs_to_eval, layout_net, k):
        ranks = []
        sample, _, _, _ = dataset[idx]
        try:
            output_list = layout_net.forward(sample[-1][1:-1], sample)
            pred = make_prediction(output_list)
        except ProcessingException:
            return None, None
        ranks.append(float(pred.cpu().numpy()))
        print('correct rank: ', ranks)
        for neg_idx in idxs_to_eval:
            distractor, _, _, _ = dataset[neg_idx]
            neg_sample = create_neg_sample(sample, distractor)
            try:
                output_list = layout_net.forward(neg_sample[-1][1:-1], neg_sample)
                pred = make_prediction(output_list)
                ranks.append(float(pred.cpu().numpy()))
            except ProcessingException:
                pass
                #np.random.seed(neg_idx)
                #ranks.append(np.random.rand(1)[0])
                #print("Adding random rank: ", ranks[-1])
        #print("all ranks: ", ranks)
        return mrr(ranks), [p_at_k(ranks, ki) for ki in k]

    layout_net.set_eval()
    results = {f'P@{ki}': [] for ki in k}
    results['MRR'] = []
    with torch.no_grad():
        np.random.seed(122)
        if count > 0 and count <= len(dataset):
            examples = np.random.choice(range(len(dataset)), count, replace=False)
        else:
            examples = np.arange(len(dataset))
        for j, ex in enumerate(examples):
            np.random.seed(ex)
            idxs = np.random.choice(range(len(dataset)), distractor_set_size+1, replace=False)
            idxs = idxs[idxs != ex]
            idxs = idxs[:distractor_set_size]
            cur_mrr, p_at_ks = get_mrr_for_one_sample(dataset, ex, idxs, layout_net, k)
            print(cur_mrr, p_at_ks)
            if cur_mrr is None or p_at_ks is None:
                continue
            results['MRR'].append(cur_mrr)
            if (j + 1) % 10 == 0:
                print(np.mean(results['MRR']))
            for ki, pre in zip(k, p_at_ks):
                results[f'P@{ki}'].append(pre)
    layout_net.set_train()
    return np.mean(results['MRR']), [np.mean(results[f'P@{ki}']) for ki in k]


def eval_acc(dataset, layout_net, make_prediction, count):
    def get_acc_for_one_sample(sample, label):
        output_list = layout_net.forward(sample[-1][1:-1], sample)
        pred = make_prediction(output_list)
        binarized_pred = binarize(pred, threshold=0.5).cpu()
        return int((binarized_pred == label).detach().numpy())

    accs = []
    layout_net.set_eval()
    positive_label = torch.tensor(1, dtype=torch.float).cpu()
    negative_label = torch.tensor(0, dtype=torch.float).cpu()
    with torch.no_grad():
        np.random.seed(122)
        if count > 0 and count <= len(dataset):
            examples = np.random.choice(range(len(dataset)), count, replace=False)
        else:
            examples = np.arange(len(dataset))
        for ex in examples:
            sample, _, _, label = dataset[ex]
            assert label == 1, 'Mismatching example sampled from dataset, but expected matching examples only'
            try:
                accs.append(get_acc_for_one_sample(sample, label=positive_label))
            except ProcessingException:
                continue
            # Create a negative example
            np.random.seed(ex)
            neg_idx = np.random.choice(range(len(dataset)), 1)[0]
            neg_sample = create_neg_sample(dataset[ex][0], dataset[neg_idx][0])
            try:
                accs.append(get_acc_for_one_sample(neg_sample, label=negative_label))
            except ProcessingException:
                continue
    layout_net.set_train()
    return np.mean(accs)


def eval_acc_f1_pretraining_task(dataset, layout_net, threshold, count, override_negatives):
    def get_acc_and_f1_for_one_sample(sample, label, threshold=threshold):
        output_list = layout_net.forward(sample[-1][1:-1], sample)
        accs = []
        f1s = []
        for out in output_list:
            true_out, pred_out = out
            if override_negatives:
                if label == 0:
                    true_out = torch.zeros_like(true_out)
            labels = binarize(true_out, threshold=threshold).cpu().detach().numpy()
            binarized_preds = binarize(pred_out, threshold=threshold).cpu().detach().numpy()
            accs.append(sum(binarized_preds == labels) * 1. / labels.shape[0])
            f1s.append(f1_score(labels, binarized_preds, zero_division=1))
        return np.mean(accs), np.mean(f1s)

    accs = []
    f1_scores = []
    layout_net.set_eval()
    with torch.no_grad():
        i = 0
        for j in range(len(dataset)):
            sample, _, _, label = dataset[j]
            assert label == 1, 'Mismatching example sampled from dataset, but expected matching examples only'
            try:
                acc, f1 = get_acc_and_f1_for_one_sample(sample, label, threshold=threshold)
                accs.append(acc)
                f1_scores.append(f1)
            except ProcessingException:
                continue
            # Create a negative example
            np.random.seed(22222 + j)
            neg_idx = np.random.choice(range(len(dataset)), 1)[0]
            neg_sample = create_neg_sample(dataset[j][0], dataset[neg_idx][0])
            try:
                acc, f1 = get_acc_and_f1_for_one_sample(neg_sample, label=0, threshold=threshold)
                accs.append(acc)
                f1_scores.append(f1)
            except ProcessingException:
                continue
            if i >= count:
                break
            i += 1
    layout_net.set_train()
    return np.mean(accs), np.mean(f1_scores)


def pretrain(layout_net, lr, adamw, checkpoint_dir, num_epochs, data_loader, clip_grad_value, device, print_every,
             writer, k, valid_data, distractor_set_size, patience, use_lr_scheduler, batch_size, skip_negatives,
             override_negatives, threshold, loss_type):
    if loss_type == 'bce_loss':
        loss_func = torch.nn.BCELoss()
    elif loss_type == 'kldiv_loss':
        loss_func = torch.nn.KLDivLoss()
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
                        true_out = torch.zeros_like(true_out)
                if loss_type == 'bce_loss':
                    labels = binarize(true_out, threshold=threshold).to(device)
                elif loss_type == 'kldiv_loss':
                    labels = true_out/torch.sum(true_out) # normalize to probability
                    pred_out = torch.logit(pred_out) #reverse sigmoid?
                elif loss_type == 'mse_loss':
                    labels = true_out
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
          make_prediction, optim_type='adam'):
    loss_func = torch.nn.BCELoss()
    if optim_type == 'sgd':
        op = torch.optim.SGD(layout_net.parameters(), lr=lr, weight_decay=adamw)
        if layout_net.weighted_cosine:
            op.add_param_group({"params": layout_net.weight})
    elif optim_type == 'adam':
        op = torch.optim.Adam(layout_net.parameters(), lr=lr, weight_decay=adamw)
        if layout_net.weighted_cosine: # try a hack?
            op.add_param_group({"params": layout_net.weight})
    else:
        raise Exception("Unknown optimizer type!! ", optim_type)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, verbose=True)
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
                op.step()
                loss = None
                for x in layout_net.parameters():
                    x.grad = None
                if layout_net.weighted_cosine:
                    layout_net.weight.grad = None
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


def main(device, data_dir, scoring_checkpoint, num_epochs, num_epochs_pretraining, lr, print_every,
         valid_file_name, num_negatives, adamw,
         example_count, dropout, checkpoint_dir, summary_writer_dir, use_lr_scheduler,
         clip_grad_value, patience, k, distractor_set_size, do_pretrain, do_train, batch_size, layout_net_training_ckp,
         finetune_scoring, override_negatives_in_pretraining, skip_negatives_in_pretraining, use_dummy_action, do_eval,
         alignment_function, pretrain_bin_threshold, pretrain_loss_type, eval_count):
    print(f"Loading dataset from {data_dir}")
    dataset = ConcatDataset([CodeSearchNetDataset_NotPrecomputed(data_dir, device), ] +
                            [CodeSearchNetDataset_NotPrecomputed_RandomNeg(filename=data_dir, device=device,
                                                                           range=r) for r in range(1)])
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

    code_in_output = False
    weighted_cosine=False
    if alignment_function == 'dot':
        make_prediction = make_prediction_dot
    elif alignment_function == 'cosine':
        make_prediction = make_prediction_cosine
    elif alignment_function == 'weighted_emb':
        make_prediction = make_prediction_weighted_embedding
        code_in_output = True
    elif alignment_function == "weighted_cosine":
        make_prediction = make_prediction_weighted_cosine_v2
        code_in_output = True
        weighted_cosine = True
    else:
        raise Exception("Unknown alignment type")
    layout_net = LayoutNet(scoring_module, action_module, device, code_in_output, weighted_cosine)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if do_pretrain:
        pretrain(layout_net=layout_net, lr=lr, adamw=adamw, checkpoint_dir=checkpoint_dir,
                 num_epochs=num_epochs_pretraining, data_loader=data_loader, clip_grad_value=clip_grad_value,
                 device=device, print_every=print_every, writer=writer, k=k, valid_data=valid_data,
                 distractor_set_size=distractor_set_size, patience=patience, use_lr_scheduler=use_lr_scheduler,
                 batch_size=batch_size, skip_negatives=skip_negatives_in_pretraining,
                 override_negatives=override_negatives_in_pretraining, threshold=pretrain_bin_threshold,
                 loss_type=pretrain_loss_type)
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
              batch_size=batch_size, make_prediction=make_prediction)
    if do_eval:
        eval(layout_net=layout_net, data=valid_data, k=k, distractor_set_size=distractor_set_size,
             count=eval_count, make_prediction=make_prediction)


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
    parser.add_argument('--eval_count', dest='eval_count', type=int, default=100,
                        help='How many examples to use in evaluation, pass -1 for evaluating on the entire validation set')

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
         eval_count=args.eval_count, )
