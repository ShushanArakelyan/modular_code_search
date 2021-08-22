import numpy as np
import bleach
import torch

from scoring.eval_finetuned_codebert import run_eval_epoch
from scoring.utils import extract_noun_tokens, get_ground_truth_matches


class TokenVal(object):
    def __init__(self, token, val):
        self.token = token
        self.val = val

    def __str__(self):
        return self.token


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def color_tokenvals(s, idx):
    r = 255 - int(s.val * 255)
    color_tuple = [r, r, r]
    color_tuple[idx] = 255
    color = rgb_to_hex(tuple(color_tuple))
    return 'background-color: %s' % color


def tokensvals_to_html(embedder, token_vals, color):
    html = '<pre style="float:left; max-width:350px; margin-right:10px"><code>'

    for i, tv in enumerate(token_vals):
        if tv.token != '<s>':
            html += '<span title="{}" style="{}">{}</span>'.format(i, color_tokenvals(tv, color), bleach.clean(
                embedder.tokenizer.convert_tokens_to_string(tv.token)))

    html += "</code></pre>"

    return html


def get_tokenvals(tokens, scores, idxs):
    token_vals = [TokenVal(c, 0) for c in tokens]
    for idx in idxs:
        token_vals[idx] = TokenVal(tokens[idx], scores[idx])
    return token_vals


def generate_HTML(title, embedder, tokens, ground_truth_idxs, scores, split_point):
    predicted_idxs = np.where(scores >= split_point)[0]

    true_scores_tvs = get_tokenvals(tokens, np.ones(len(tokens)), ground_truth_idxs)
    predicted_scores_tvs = get_tokenvals(tokens, scores, predicted_idxs)
    return f'<h2>{title}</h2>{0}{1}'.format(tokensvals_to_html(embedder, predicted_scores_tvs, 0),
                                            tokensvals_to_html(embedder, true_scores_tvs, 1))


def find_split_point(data, scorer, embedder):
    avg_f1 = []
    split_points = np.arange(0, 1, 0.05)
    for i in split_points:
        f1_scores, precisions, recalls = run_eval_epoch(data, scorer, embedder, split_point=i)
        avg_f1.append(np.mean(f1_scores))
    return split_points, avg_f1


def compute_f1(ground_truth_idxs, scores, split_point):
    predicted_idxs = np.where(scores >= split_point)[0]
    S_g = len(ground_truth_idxs)
    S_a = len(predicted_idxs)
    intersection = len(np.intersect1d(predicted_idxs, ground_truth_idxs))
    if S_g == 0:
        f1_score = 1 if S_a == 0 else 0
        return f1_score, 1 if S_a == 0 else 0, 1 if S_a == 0 else 0
    if S_a == 0:
        P_t = 0
    else:
        P_t = intersection / S_a
    R_t = intersection / S_g
    if P_t == 0 and R_t == 0:
        f1_score = 0
    else:
        f1_score = (2 * P_t * R_t) / (P_t + R_t)
    return f1_score, P_t, R_t


def eval_example(data, it, scorer, embedder, evaluate, split_point=0.5):
    result_dict = {}
    if evaluate == "F1":
        result_dict['f1_scores_for_sample'] = []
        result_dict['pre_for_sample'] = []
        result_dict['re_for_sample'] = []
    elif evaluate == "HTML":
        result_dict['html'] = []
    else:
        raise Exception("Invalid value of evaluate parameter")

    doc = data['docstring_tokens'][it]
    code = data['alt_code_tokens'][it]
    static_tags = data['static_tags'][it]
    regex_tags = data['regex_tags'][it]
    noun_tokens = extract_noun_tokens(' '.join(doc))

    out_tuple = embedder.embed_and_filter(doc, code, noun_tokens)
    noun_token_id_mapping, noun_token_embeddings, code_token_id_mapping, code_embedding, _, truncated_code_tokens = out_tuple
    for nti, nte, nt in zip(noun_token_id_mapping, noun_token_embeddings, noun_tokens):
        nte = nte.unsqueeze(0)
        # check for regex and static matches
        ground_truth_idxs = get_ground_truth_matches(nt, code, code_token_id_mapping, static_tags, regex_tags)
        # forward pass through scorer
        tiled_nte = nte.repeat(len(truncated_code_tokens), 1)
        forward_input = torch.cat((tiled_nte, code_embedding), dim=1)
        scorer_out = torch.sigmoid(scorer.forward(forward_input)).squeeze().cpu().detach().numpy()
        if evaluate == "F1":
            f1, p, re = compute_f1(ground_truth_idxs, scorer_out, split_point)
            result_dict['f1_scores_for_sample'].append(f1)
            result_dict['pre_for_sample'].append(p)
            result_dict['re_for_sample'].append(re)
        elif evaluate == "HTML":
            result_dict['html'].append(
                generate_HTML(nt, embedder, truncated_code_tokens, ground_truth_idxs, scorer_out, split_point))
    return result_dict
