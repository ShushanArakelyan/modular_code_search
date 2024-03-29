import re

import numpy as np


def get_noun_phrases(ccg_parse):
    ccg_parse = ccg_parse[1:-1]  # remove brackets from the beginning and the end
    ccg_parse = ccg_parse.replace('(', '( ')
    # remove '@Concat' operation
    i = 0
    parts = ccg_parse.split(' ')
    modified_parse = ''
    while i < len(parts):
        if parts[i] == '':
            i += 1
            continue
        if parts[i] == '@Concat':
            i += 2
        modified_parse += ' ' + parts[i]
        i += 1
    # extract consecutive noun phrases
    i = 0
    parts = modified_parse.split(' ')
    phrases = []
    while i < len(parts):
        phrase = []
        while i < len(parts) and not (parts[i].startswith('@') or parts[i].startswith(')')):
            if len(parts[i]) != 0:
                phrase.append(parts[i])
            i += 1
        if len(phrase) != 0:
            phrases.append(phrase)
        if i == len(parts):
            break
        if parts[i].startswith('@Action'):
            i += 3
        elif parts[i].startswith('@'):
            i += 2
        else:
            i += 1
    return phrases


def embed_pair(embedder, phrase, code, embed_separately):
    if not embed_separately:
        embedder_out = embedder.embed(phrase, code)
        if embedder_out is None:
            return None
        if embedder_out[0].size == 0 or embedder_out[2].size == 0:
            return None
    else:
        phrase_embedder_out = embedder.embed(phrase, [' '])
        code_embedder_out = embedder.embed([' '], code)
        if phrase_embedder_out is None or code_embedder_out is None:
            return None
        word_token_id_mapping, word_token_embeddings, _, __, ___, ____, cls_token_embedding = phrase_embedder_out
        _, __, code_token_id_mapping, code_embedding, _, truncated_code_tokens, ___ = code_embedder_out
        embedder_out = (word_token_id_mapping, word_token_embeddings, code_token_id_mapping, code_embedding, 'None',
                        truncated_code_tokens, cls_token_embedding)
        if word_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
            return None
    return embedder_out


def get_matched_labels_binary_v2(tokens, code_token_id_mapping, query_token):
    tags = np.zeros(max(code_token_id_mapping[-1]) + 1)
    for i, t in enumerate(tokens):
        if re.search(query_token.lower(), t.lower()):
            tags[code_token_id_mapping[i]] = 1
    return np.asarray(tags)


def get_static_labels_binary(code_token_id_mapping, query_token, static_tags):
    query_token = query_token.lower()
    static_labels = {
        'list': ['AstList', 'AstListComp'],
        'dict': ['AstDict', 'AstDictComp'],
        'generator': ['AstGen'],
        'set': ['AstSet', 'AstSetComp'],
        'bool': ['AstBoolOp'],
        'char': ['AstChar'],
        'num': ['AstNum'],
        'str': ['AstStr'],
        'tuple': ['AstTuple'],
        'compare': ['AstCompare']}
    is_datatype = [label for label in static_labels if label in query_token]
    static_match = np.zeros(max(code_token_id_mapping[-1]) + 1)
    for datatype in is_datatype:
        labels = static_labels[datatype]
        # because the data has been truncated, iterate over the truncated version
        for i in range(len(code_token_id_mapping)):
            tags = static_tags[i]
            exists = [label in tag for label in labels for tag in tags]
            if any(exists):
                for idx in code_token_id_mapping[i]:
                    static_match[idx] = 1
    return static_match


def get_regex_labels_binary(code_token_id_mapping, query_token, regex_tags):
    query_token = query_token.lower()
    datatype_regex_matches = {'dict': ['dict', 'map'],
                              'list': ['list', 'arr'],
                              'tuple': ['tuple'],
                              'int': ['count', 'cnt', 'integer'],
                              'file': ['file'],
                              'enum': ['enum'],
                              'string': ['str', 'char', 'unicode', 'ascii'],
                              'path': ['path', 'dir'],
                              'bool': ['true', 'false', 'bool'],
                              }
    is_datatype = [datatype for datatype in datatype_regex_matches for regex_match in datatype_regex_matches[datatype]
                   if regex_match in query_token]
    regex_match = np.zeros(max(code_token_id_mapping[-1]) + 1)
    for datatype in is_datatype:
        # because the data has been truncated, iterate over the truncated version
        for i in range(len(code_token_id_mapping)):
            tags = regex_tags[i]
            if datatype in tags:
                for idx in code_token_id_mapping[i]:
                    regex_match[idx] = 1
    return regex_match


def get_ground_truth_matches(query_token, code, code_token_id_mapping, static_tags, regex_tags):
    ground_truth_idxs = []
    idxs = get_matched_labels_binary_v2(code[:len(code_token_id_mapping)],
                                        code_token_id_mapping, query_token).nonzero()[0]
    if idxs.size:
        ground_truth_idxs.extend(idxs)
    idxs = get_static_labels_binary(code_token_id_mapping, query_token, static_tags).nonzero()[0]
    if idxs.size:
        ground_truth_idxs.extend(idxs)
    idxs = get_regex_labels_binary(code_token_id_mapping, query_token, regex_tags).nonzero()[0]
    if idxs.size:
        ground_truth_idxs.extend(idxs)
    return ground_truth_idxs
