import argparse
import sys

import numpy as np
import torch
import tqdm


def compute_mrr(ranks):
    return 1. / (np.sum(ranks >= ranks[0]))


def compute_p_at_k(ranks, k):
    return 1 if np.sum(ranks >= ranks[0]) <= k else 0


def create_neg_sample(orig, distr):
    return orig[0], distr[1], distr[2], distr[3], orig[4]


def make_prediction_layout_net(layout_net, dataset, orig_idx, distractor_idx=None):
    sample, _, _, _ = dataset[orig_idx]
    if distractor_idx is not None:
        distractor, _, _, _ = dataset[distractor_idx]
        sample = create_neg_sample(sample, distractor)
    pred = layout_net.forward(sample[-1][1:-1], sample)
    if pred is None:
        np.random.seed(orig_idx)
        return np.random.uniform()
    return float(torch.sigmoid(pred[0][1]).cpu().numpy())


def make_prediction_codebert(codebert, dataset, orig_idx, distractor_idx=None):
    if distractor_idx is None:
        distractor_idx = orig_idx
    sample, _, _, _ = dataset[orig_idx]
    distractor_sample, _, _, _ = dataset[distractor_idx]
    inputs = codebert.get_feature_inputs(' '.join(sample[0]),
                                         ' '.join(distractor_sample[1]))
    output = codebert.classifier(**inputs)
    return float(torch.sigmoid(output['logits'])[0][1].cpu().numpy())


def read_distractors_from_file(orig_idx):
    raise NotImplementedError()


def generate_random_distractors(dataset, orig_idx, n=1000):
    np.random.seed(orig_idx * 3 + 5)
    distractors = np.random.choice(range(len(dataset)), min(n, len(dataset)), replace=False)
    distractors = distractors[distractors != orig_idx][:n - 1]
    return distractors


def eval_against_distractors(dataset, orig_idx, distr_set, model, make_prediction):
    ranks = [make_prediction(model, dataset, orig_idx)]
    for distr_idx in distr_set:
        ranks.append(make_prediction(model, dataset, orig_idx, distr_idx))
    ranks = np.asarray(ranks)
    return ranks


def write_to_file(results, output_file):
    output_file.write(' '.join([f"{key}: {round(np.mean(values), 4)}, " for key, values in results.items()]))
    output_file.write('\n')
    output_file.flush()


def update_results(results, mrr, k, p_at_k):
    results["mrr"].append(mrr)
    for i, ki in enumerate(k):
        results[f"P@{ki}"].append(p_at_k[i])
    return results


def eval(dataset, model, make_prediction, distractor_generator, distractor_count, k, output_file=None):
    if isinstance(k, int):
        k = [k]
    results = {f"P@{ki}": [] for ki in k}
    results["mrr"] = []
    for i in tqdm.tqdm(range(len(dataset))):
        distr_set = distractor_generator(dataset, i, distractor_count)
        ranks = eval_against_distractors(dataset, i, distr_set, model, make_prediction)
        mrr = compute_mrr(ranks)
        p_at_k = tuple(compute_p_at_k(ranks, ki) for ki in k)
        results = update_results(results, mrr, k, p_at_k)
        if output_file:
            write_to_file(results, output_file)
    return results


def staged_eval(dataset, model_st1, model_st2, make_prediction_st1, make_prediction_st2, distractor_generator,
                distractor_count_st1, distractor_count_st2, k, output_file=None):
    if isinstance(k, int):
        k = [k]
    results_st1 = {f"P@{ki}": [] for ki in k}
    results_st1["mrr"] = []
    results_st2 = {f"P@{ki}": [] for ki in k}
    results_st2["mrr"] = []
    for i in tqdm.tqdm(range(len(dataset))):
        distr_set = distractor_generator(dataset, i, distractor_count_st1)
        ranks = eval_against_distractors(dataset, i, distr_set, model_st1, make_prediction_st1)
        top_distractors = np.argsort(ranks)[::-1][:distractor_count_st2] - 1
        if -1 in top_distractors:
            top_distractors = top_distractors[top_distractors != -1][:distractor_count_st2 - 1]
        mrr = compute_mrr(ranks)
        p_at_k = tuple(compute_p_at_k(ranks, ki) for ki in k)
        results_st1 = update_results(results_st1, mrr, k, p_at_k)
        ranks = eval_against_distractors(dataset, i, top_distractors, model_st2, make_prediction_st2)
        mrr = compute_mrr(ranks)
        p_at_k = tuple(compute_p_at_k(ranks, ki) for ki in k)
        results_st2 = update_results(results_st2, mrr, k, p_at_k)
        if output_file:
            write_to_file(results_st1, output_file)
            write_to_file(results_st2, output_file)
    return results_st1, results_st2


def main(args):
    from eval.dataset import CodeSearchNetDataset_NotPrecomputed
    from layout_assembly.layout_codebert_classifier import LayoutNet_w_codebert_classifier as LayoutNet
    from layout_assembly.modules import ActionModuleFacade, ScoringModule

    dataset = CodeSearchNetDataset_NotPrecomputed(args.valid_file_name, device=args.device)

    if args.distractor_type == 'staged':
        assert len(args.model_type) == len(args.model_checkpoint)
        assert len(args.model_type) == len(args.distractor_count)
    else:
        assert len(args.model_type) == 1
        assert len(args.model_checkpoint) == 1
        assert len(args.distractor_count) == 1

    models = []
    make_prediction_funcs = []
    for i, model_type in enumerate(args.model_type):
        if model_type == 'layout_net':
            scoring_module = ScoringModule(args.device, args.scoring_checkpoint)
            action_module = ActionModuleFacade(args.device, version=args.version, normalized=True, dropout=0.2)
            model = LayoutNet(scoring_module, action_module, args.device,
                              precomputed_scores_provided=False,
                              use_cls_for_verb_emb=args.use_cls_for_verb_emb,
                              use_constant_for_weights=args.use_constants_for_weights)
            model.load_from_checkpoint(args.model_checkpoint[i])
            model.set_eval()
            make_prediction = make_prediction_layout_net
            models.append(model)
            make_prediction_funcs.append(make_prediction)
        elif model_type == 'codebert':
            import codebert_embedder as model
            from transformers import RobertaForSequenceClassification
            model.init_embedder(args.device)
            model.classifier = RobertaForSequenceClassification.from_pretrained(args.model_checkpoint[i]).to(args.device)
            make_prediction = make_prediction_codebert
            models.append(model)
            make_prediction_funcs.append(make_prediction)
        else:
            raise Exception("Unknown model type")

    if args.distractor_type == "codebert":
        raise NotImplementedError("Using top 10 distractors from codebert is not implemented")
    get_distractors = generate_random_distractors
    with open(args.output_file, 'w') as out_file:
        with torch.no_grad():
            if args.distractor_type != 'staged':
                eval(dataset, models[0], make_prediction_funcs[0],
                     get_distractors, args.distractor_count[0], args.k, out_file)
            else:
                staged_eval(dataset, models[0], models[1], make_prediction_funcs[0], make_prediction_funcs[1],
                            get_distractors, args.distractor_count[0], args.distractor_count[1], args.k, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='End-to-end training of neural module network')
    parser.add_argument('--device', dest='device', type=str, help='device to run on')
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str,
                        help='Scoring module checkpoint')
    parser.add_argument('--model_type', dest='model_type', type=str, action='append',
                        help='Can be "layout_net" or "codebert"')
    parser.add_argument('--distractor_type', dest='distractor_type', type=str,
                        help='Can be "random","staged", or "top-codebert"')
    parser.add_argument('--distractor_count', dest='distractor_count', type=int, action='append',
                        help='Number of distractors')
    parser.add_argument('--version', dest='version', type=int,
                        help='Action version')
    parser.add_argument('--use_cls_for_verb_emb', dest='use_cls_for_verb_emb',
                        default=False, action='store_true')
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='Validation data file', required=True)
    parser.add_argument('--model_checkpoint', dest='model_checkpoint', type=str, action='append',
                        help='Continue training from this checkpoint, not implemented')
    parser.add_argument('--output_file', dest='output_file', type=str,
                        help='File to store the output of the evaluation')
    parser.add_argument('--project_dir', dest='project_dir')
    parser.add_argument('--use_constants_for_weights', dest='use_constants_for_weights',
                        default=False, action='store_true')
    parser.add_argument('--k', action='append', type=int)
    args = parser.parse_args()
    sys.path.append(args.project_dir)
    main(args)
