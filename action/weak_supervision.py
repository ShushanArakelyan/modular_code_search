import torch
import numpy as np
import re

from .ws_regex_dict import REGEX_DICT

def supervision_scores(sample, scores, verb):
    reg_scores = []
    norm_scores = []
    for r, n in scores:
        reg_scores.append(r)
        norm_scores.append(n)
    new_scores = torch.zeros_like(scores)
    return new_scores


def regex_matching(verb, line, scores):
    matched = False
    for r in REGEX_DICT[verb]:
        matches = re.search(r, line)
        if matches is not None:
            matched = True
    if matched:
        return 1
    return 0


def random(verb, line, scores):
    if np.random.rand(1) > 0.5:
        return 1
    return 0


def uniform(verb, line, scores):
    return 1


def propagate(verb, line, scores):
    # pooling
    # return pooled scores
    raise NotImplementedError()


def weak_supervision_scores(embedder, code, verb, attend_scores, matching_func):
    code = ' '.join(code).split('\n')
    scores_per_line = [0]
    for line in code[1:]:
        scores_per_line.append(matching_func(verb, line, attend_scores))
    return scores_per_line_to_scores_per_token(embedder, code, scores_per_line)


def scores_per_line_to_scores_per_token(embedder, loc, scores_per_line):
    scores_per_token = []
    N = len(loc)
    for i, (score, line) in enumerate(zip(scores_per_line, loc)):
        if i != N - 1:
            line += '\n'
        tokens_per_line = embedder.tokenizer.tokenize(line)
        # print("ws tokens: ", tokens_per_line)
        # print("len of tokens in line: ", len(tokens_per_line))
        scores_per_token.extend([score] * len(tokens_per_line))
    return scores_per_token
