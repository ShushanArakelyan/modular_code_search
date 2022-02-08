import numpy as np
import torch
from sklearn.metrics import f1_score

from eval.utils import mrr, p_at_k
from layout_assembly.utils import ProcessingException


def create_neg_sample(orig, distr):
    return (orig[0], distr[1], distr[2], distr[3], orig[4])


def binarize(a, threshold):
    with torch.no_grad():
        return torch.where(a < threshold, torch.zeros_like(a), torch.ones_like(a))


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


def make_prediction_weighted_cosine_v3(output_list):
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
        # norm_s = 0.125 * torch.pow(s + 1, 3)
        norm_s = torch.sigmoid(s)
        if alignment_scores is None:
            alignment_scores = norm_s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, norm_s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_weighted_cosine_v4(output_list):
    alignment_scores = None
    cos = torch.nn.CosineSimilarity(dim=0)
    output_list, v = output_list
    for i in range(len(output_list)):
        a, b, code = output_list[i]
        dot_score = torch.dot(a.squeeze(), b.squeeze())
        dot_score /= sum(a).squeeze()

        N = min(a.shape[0], b.shape[0], code.shape[0])
        a_norm = a / torch.sum(a[:N, :])
        b_norm = b / torch.sum(b[:N, :])
        weighted_code_a = torch.mm(a_norm[:N, :].T, code[:N, :]).squeeze()
        weighted_a = v.squeeze() * weighted_code_a
        weighted_code_b = torch.mm(b_norm[:N, :].T, code[:N, :]).squeeze()
        weighted_b = v.squeeze() * weighted_code_b
        s = cos(weighted_a, weighted_b)
        # norm_s = 0.125 * torch.pow(s + 1, 3)
        norm_s = torch.sigmoid(s)
        final_score = dot_score * norm_s
        if alignment_scores is None:
            alignment_scores = final_score.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, final_score.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_weighted_cosine_v5(output_list):
    alignment_scores = None
    cos = torch.nn.CosineSimilarity(dim=0)
    output_list, v = output_list
    for i in range(len(output_list)):
        a, b = output_list[i]

        N = min(a.shape[0], b.shape[0])
        a_norm = a[:N, :]
        b_norm = b[:N, :]
        v_norm = v[:N, :].squeeze()
        weighted_a = v_norm * a_norm.squeeze()
        weighted_b = v_norm * b_norm.squeeze()
        s = cos(weighted_a, weighted_b)
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_kldiv(output_list):
    alignment_scores = None
    for i in range(len(output_list)):
        a, b = output_list[i]
        N = min(a.shape[0], b.shape[0])
        a_norm = a[:N, :] / torch.sum(a[:N, :])
        b_norm = b[:N, :] / torch.sum(b[:N, :])

        kldiv = torch.nn.functional.kl_div(a_norm.squeeze(), b_norm.squeeze(), reduction='batchmean')
        final_score = 1. / (1. + kldiv)
        if alignment_scores is None:
            alignment_scores = final_score.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, final_score.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_l2norm(output_list):
    alignment_scores = None
    for i in range(len(output_list)):
        a, b = output_list[i]
        N = min(a.shape[0], b.shape[0])
        truncated_a = a[:N, :].squeeze()
        truncated_b = b[:N, :].squeeze()
        l2norm = torch.linalg.norm(truncated_a - truncated_b)
        final_score = 1./(1. + l2norm)
        if alignment_scores is None:
            alignment_scores = final_score.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, final_score.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_l2norm_weighted(output_list):
    alignment_scores = None
    for i in range(len(output_list)):
        a, b, code = output_list[i]
        N = min(a.shape[0], b.shape[0], code.shape[0])
        truncated_a = a[:N, :]
        truncated_b = b[:N, :]
        truncated_code = code[:N, :]
        weighted_code_a = torch.mm(truncated_a.T, truncated_code).squeeze()
        weighted_code_b = torch.mm(truncated_b.T, truncated_code).squeeze()
        l2norm = torch.linalg.norm(weighted_code_a - weighted_code_b)
        final_score = 1./(1. + l2norm)
        if alignment_scores is None:
            alignment_scores = final_score.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, final_score.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_mlp(output_list):
    alignment_scores = None
    output_list, mlp = output_list
    for i in range(len(output_list)):
        a, b, code = output_list[i]
        N = min(a.shape[0], b.shape[0], code.shape[0])
        a_norm = a / torch.sum(a[:N, :])
        b_norm = b / torch.sum(b[:N, :])
        weighted_code_a = torch.mm(a_norm[:N, :].T, code[:N, :])
        weighted_code_b = torch.mm(b_norm[:N, :].T, code[:N, :])
        input = torch.cat((weighted_code_a, weighted_code_b), dim=1)
        final_score = mlp(input)
        if alignment_scores is None:
            alignment_scores = final_score.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, final_score.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_dot(output_list):
    # normalize by scoring module vector
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


def make_prediction_dot_v2(output_list):
    # normalize by action module vector
    alignment_scores = None
    for i in range(len(output_list)):
        s = torch.dot(output_list[i][0].squeeze(), output_list[i][1].squeeze())
        s /= sum(output_list[i][1]).squeeze()
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_dot_v3(output_list):
    # normalize by both scoring and action module vectors
    alignment_scores = None
    for i in range(len(output_list)):
        s = torch.dot(output_list[i][0].squeeze(), output_list[i][1].squeeze())
        s /= (sum(output_list[i][0]).squeeze() + sum(output_list[i][1]).squeeze())
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_dot_v4(output_list):
    # do not normalize
    alignment_scores = None
    for i in range(len(output_list)):
        s = torch.dot(output_list[i][0].squeeze(), output_list[i][1].squeeze())
        s = torch.sigmoid(s)
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_dot_v5(output_list):
    # normalize with softmax
    alignment_scores = None
    for i in range(len(output_list)):
        a = torch.softmax(output_list[i][0].squeeze())
        b = torch.softmax(output_list[i][1].squeeze())
        s = torch.dot(a, b)
        s = torch.sigmoid(s)
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def make_prediction_dot_v6(output_list):
    # normalize both v2
    alignment_scores = None
    for i in range(len(output_list)):
        a = output_list[i][0].squeeze()/torch.sum(a)
        b = output_list[i][1].squeeze()/torch.sum(b)
        s = torch.dot(a, b)
        s = torch.sigmoid(s)
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
        # s = (s + 1) * 0.5 # our scores are all non-negative, so cosine will be positive
        if alignment_scores is None:
            alignment_scores = s.unsqueeze(0)
        else:
            alignment_scores = torch.cat((alignment_scores, s.unsqueeze(dim=0)))
    pred = torch.prod(alignment_scores)
    return pred


def get_alignment_function(alignment_function):
    code_in_output = False
    weighted_cosine = False
    weighted_cosine_v2 = False
    mlp_prediction = False
    if alignment_function == 'dot':
        make_prediction = make_prediction_dot
    elif alignment_function == 'dot_v2':
        make_prediction = make_prediction_dot_v2
    elif alignment_function == 'dot_v3':
        make_prediction = make_prediction_dot_v3
    elif alignment_function == 'dot_v4':
        make_prediction = make_prediction_dot_v4
    elif alignment_function == 'cosine':
        make_prediction = make_prediction_cosine
    elif alignment_function == 'weighted_emb':
        make_prediction = make_prediction_weighted_embedding
        code_in_output = True
    elif alignment_function == "weighted_cosine_v1":
        make_prediction = make_prediction_weighted_cosine
        code_in_output = True
        weighted_cosine = True
    elif alignment_function == "weighted_cosine_v2":
        make_prediction = make_prediction_weighted_cosine_v2
        code_in_output = True
        weighted_cosine = True
    elif alignment_function == "weighted_cosine_v3":
        make_prediction = make_prediction_weighted_cosine_v3
        code_in_output = True
        weighted_cosine = True
    elif alignment_function == "weighted_cosine_v4":
        make_prediction = make_prediction_weighted_cosine_v4
        code_in_output = True
        weighted_cosine = True
    elif alignment_function == "weighted_cosine_v5":
        make_prediction = make_prediction_weighted_cosine_v5
        weighted_cosine_v2 = True
    elif alignment_function == "kldiv":
        make_prediction = make_prediction_kldiv
    elif alignment_function == "l2norm":
        make_prediction = make_prediction_l2norm
    elif alignment_function == "l2norm_weighted":
        code_in_output = True
        make_prediction = make_prediction_l2norm_weighted
    elif alignment_function == "mlp":
        make_prediction = make_prediction_mlp
        code_in_output = True
        mlp_prediction = True
    else:
        raise Exception("Unknown alignment type")
    return code_in_output, weighted_cosine, weighted_cosine_v2, mlp_prediction, make_prediction



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
