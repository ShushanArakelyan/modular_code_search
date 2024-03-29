import bleach
import numpy as np
import torch

from scoring.utils import get_noun_phrases, get_ground_truth_matches, embed_pair


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
    color_tuple = np.asarray([r, r, r])
    color_tuple[idx] = 255
    color = rgb_to_hex(tuple(color_tuple))
    return 'background-color: %s' % color


def tokenvals_to_html(embedder, token_vals, color):
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


def generate_HTML(title, embedder, tokens, ground_truth_idxs, scores, split_point, color):
    predicted_idxs = np.where(scores >= split_point)[0]

    true_scores_tvs = get_tokenvals(tokens, np.ones(len(tokens)), ground_truth_idxs)
    predicted_scores_tvs = get_tokenvals(tokens, scores, predicted_idxs)
    return '<h2>{0}</h2>{1}{2}'.format(title, tokenvals_to_html(embedder, predicted_scores_tvs, color),
                                       tokenvals_to_html(embedder, true_scores_tvs, 1))


def normalize_predictions(scorer_out, scorer, embedder, code, embed_separately, version):
    random_words = ['gate', 'version', 'average', 'logic']
    scores_for_random_words = []
    for word in random_words:
        out_tuple = embed_pair(embedder, word, code, embed_separately)
        scores = scorer_forward(scorer, out_tuple, version)
        scores_for_random_words.append(scores)
    scores_for_random_words = np.hstack([np.expand_dims(scores, 1) for scores in scores_for_random_words])
    new_scores = (scorer_out - np.mean(scores_for_random_words, axis=1))
    max_scores = np.max(np.hstack([scores_for_random_words, np.expand_dims(scorer_out, 1)]), axis=1)
    min_scores = np.min(np.hstack([scores_for_random_words, np.expand_dims(scorer_out, 1)]), axis=1)
    new_scores /= (max_scores - min_scores)
    new_scores = np.clip(new_scores, 0, 1)
    return new_scores


def scorer_forward(scorer, embedder_out, version='CLS'):
    _, word_token_embs, code_token_id_mapping, code_embs, __, truncated_code_tokens, cls_token_emb = embedder_out
    if version == 'CLS':
        emb = cls_token_emb
    elif version == 'MEAN':
        emb = torch.mean(word_token_embs, dim=0, keepdim=True)
    else:
        raise Exception("Unknown average computing version!")
    tiled_emb = emb.repeat(len(truncated_code_tokens), 1)
    forward_input = torch.cat((tiled_emb, code_embs), dim=1)
    token_count = max(code_token_id_mapping[-1])
    scorer_out = torch.sigmoid(scorer.forward(forward_input)).squeeze().cpu().detach().numpy()[:token_count]
    return scorer_out


def compute_f1(ground_truth_idxs, predicted_idxs):
    S_g = len(ground_truth_idxs)
    S_a = len(predicted_idxs)
    intersection = len(np.intersect1d(predicted_idxs, ground_truth_idxs))
    if S_g == 0:
        return None
    if S_a == 0:
        return 0, 0, 0
    P_t = intersection / S_a
    R_t = intersection / S_g
    if P_t == 0 and R_t == 0:
        f1_score = 0
    else:
        f1_score = (2 * P_t * R_t) / (P_t + R_t)
    return f1_score, P_t, R_t


def eval_example(data, it, scorer, embedder, evaluate, embed_separately=False, split_point=0.5, version="CLS",
                 normalize=False, color=[0]):
    result_dict = {}
    if evaluate == "F1":
        result_dict['f1_scores_for_sample'] = []
        result_dict['pre_for_sample'] = []
        result_dict['re_for_sample'] = []
    elif evaluate == "HTML":
        result_dict['html'] = []
    else:
        raise Exception("Invalid value of evaluate parameter")

    code = data['alt_code_tokens'][it]
    static_tags = data['static_tags'][it]
    regex_tags = data['regex_tags'][it]
    ccg_parse = data['ccg_parse'][it]

    if len(ccg_parse) == 0 or '\\' in ccg_parse:
        # TODO: this example is not parsed properly, skip for now, but handle somehow in the future
        return result_dict

    phrases = get_noun_phrases(ccg_parse)
    for phrase in phrases:
        out_tuple = embed_pair(embedder, phrase, code, embed_separately)
        if out_tuple is None:
            if evaluate == "F1":
                result_dict['f1_scores_for_sample'] = []
                result_dict['pre_for_sample'] = []
                result_dict['re_for_sample'] = []
            elif evaluate == "HTML":
                result_dict['html'] = []
            return result_dict
        _, noun_token_embs, code_token_id_mapping, code_embs, __, truncated_code_tokens, cls_token_emb = out_tuple
        ground_truth_idxs_for_phrase = []
        for token in phrase:
            ground_truth_idxs = get_ground_truth_matches(token, code, code_token_id_mapping, static_tags, regex_tags)
            ground_truth_idxs_for_phrase.extend(ground_truth_idxs)
        ground_truth_idxs_for_phrase = np.unique(ground_truth_idxs_for_phrase)
        scorer_out = scorer_forward(scorer, out_tuple, version)
        if normalize:
            scorer_out = normalize_predictions(scorer_out, scorer, embedder, code, embed_separately, version)
        predicted_idxs = np.where(scorer_out > split_point)[0]
        if evaluate == "F1":
            # this is equivalent to maxpooling
            reversed_code_token_id_mapping = {ai: i for i, a in enumerate(code_token_id_mapping) for ai in a}
            orig_token_predicted_idx = np.unique([reversed_code_token_id_mapping[idx] for idx in predicted_idxs])
            orig_token_ground_truth_idx = np.unique(
                [reversed_code_token_id_mapping[idx] for idx in ground_truth_idxs_for_phrase])
            out = compute_f1(orig_token_ground_truth_idx, orig_token_predicted_idx)
            if out is None:
                continue
            else:
                f1, p, re = out
                result_dict['f1_scores_for_sample'].append(f1)
                result_dict['pre_for_sample'].append(p)
                result_dict['re_for_sample'].append(re)
        elif evaluate == "HTML":
            result_dict['html'].append(
                generate_HTML(' '.join(phrase), embedder, truncated_code_tokens, ground_truth_idxs_for_phrase,
                              scorer_out,
                              split_point, color))
    return result_dict


def find_split_point(data, scorer, embedder, embed_separately, version, normalize):
    avg_f1 = []
    split_points = np.arange(0, 1, 0.05)
    for i in split_points:
        f1_scores = []
        precisions = []
        recalls = []
        for it in range(len(data)):
            result_dict = eval_example(data, it, scorer, embedder, evaluate="F1", split_point=i,
                                       embed_separately=embed_separately, version=version, normalize=normalize)
            f1 = result_dict['f1_scores_for_sample']
            pre = result_dict['pre_for_sample']
            re = result_dict['re_for_sample']
            if len(f1) > 0 and len(pre) > 0 and len(re) > 0:
                f1_scores.append(np.mean(f1))
                precisions.append(np.mean(pre))
                recalls.append(np.mean(re))
        if len(f1_scores) > 0:
            avg_f1.append(np.mean(f1_scores))
        else:
            avg_f1.append(0)
    return split_points, avg_f1
