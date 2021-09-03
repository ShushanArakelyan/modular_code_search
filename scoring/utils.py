import re

import numpy as np
import spacy
nlp = spacy.load("en_core_web_md")


def extract_noun_tokens(doc):
    """Having the docstring, the function returns only the word that are nouns."""
    tokens = nlp(doc)
    pos_tags = [token.tag_ for token in tokens]
    # checks whether the word has expected pos tag
    noun_tokens = []
    for i, pos in enumerate(pos_tags):
        if pos.startswith('NN') and tokens[i].text.isalnum():
            # we lower the word, as many words are recognized by tokenizer when they are lowered
            noun_tokens.append(tokens[i].text.lower())
    return noun_tokens


def get_noun_phrases(ccg_parse):
    ccg_parse = ccg_parse[1:-1] # remove brackets from the beginning and the end
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
            if len(parts[i])  != 0:
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


def get_matched_labels_binary(code_token_id_mapping, query_token, nlp_cache):
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    matcher.add(query_token, [[{"TEXT": {"REGEX": query_token}}]])

    matches = [matcher(nlp_t) for nlp_t in nlp_cache]
    tags = [1 if match else 0 for i, match in enumerate(matches) for _ in code_token_id_mapping[i]]
    return np.asarray(tags)


def get_matched_labels_binary_v2(tokens, code_token_id_mapping, query_token):
#     lemmatizer = WordNetLemmatizer()
    tags = np.zeros(max(code_token_id_mapping[-1]) + 1)
    for i, t in enumerate(tokens):
        if re.search(query_token.lower(), t.lower()):
            tags[code_token_id_mapping[i]] = 1
#         elif re.search(lemmatizer.lemmatize(query_token.lower()), t.lower()):
#             tags[code_token_id_mapping[i]] = 1
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