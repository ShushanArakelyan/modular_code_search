import numpy as np


def mrr(rs):
    correct_score = rs[0]
    scores = np.array(rs[1:])
    rank = np.sum(scores >= correct_score) + 1
    return np.mean(1.0 / rank)


def p_at_k(rs, k):
    correct_score = rs[0]
    scores = np.array(rs[1:])
    rank = np.sum(scores >= correct_score) + 1
    return 1 if rank <= k else 0


def mrr_distance(rs):
    correct_score = rs[0]
    scores = np.array(rs[1:])
    rank = np.sum(scores < correct_score) + 1
    return np.mean(1.0 / rank)
