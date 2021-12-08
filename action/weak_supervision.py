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


def regex_supervision(code, verb):
    scores_per_line = [0]
    for line in code[1:]:
        matched = False
        for r in REGEX_DICT[verb]:
            matches = re.search(r, line)
            if matches is not None:
                matched = True
        if matched:
            scores_per_line.append(1)
        else:
            scores_per_line.append(0)
    return scores_per_line


def scores_per_line_to_scores_per_token(embedder, loc, scores_per_line):
    scores_per_token = []
    for score, line in zip(loc, scores_per_line):
        tokens_per_line = embedder.tokenizer.tokenize(line)
        scores_per_token.extend([score] * len(tokens_per_line))
    return scores_per_token
