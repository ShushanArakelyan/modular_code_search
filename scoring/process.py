import numpy as np
import pandas as pd
import spacy
import sys
import torch
import re

from tqdm import tqdm

from .embedder import Embedder

nlp = spacy.load("en_core_web_md")
P = 0.7 # probability of sampling a negative example


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


def filter_embedding_by_id(query_embedding, noun_token_ids):
    noun_token_embeddings = []
    for nti in noun_token_ids:
        nte = []
        for i in nti:
            nte.append(query_embedding[i])
        nte = np.mean(nte, 0)
        noun_token_embeddings.append(nte)
    return np.asarray(noun_token_embeddings)


def get_word_to_roberta_tokens(orig_tokens, codebert_tokens, noun_tokens, embedder):
    rt_i = 0
    nt_i = 0
    last_end = 0
    output_tokens = []
    while rt_i < len(codebert_tokens) and nt_i < len(orig_tokens):
        if embedder.tokenizer.convert_tokens_to_string(codebert_tokens[:rt_i + 1]) == " ".join(orig_tokens[:nt_i + 1]):
            current_token_idxs = np.arange(last_end, rt_i + 1)
            if orig_tokens[nt_i] in noun_tokens:
                output_tokens.append(current_token_idxs)
            last_end = rt_i + 1
            nt_i += 1
        rt_i += 1
    return output_tokens


def get_code_to_roberta_tokens(orig_tokens, codebert_tokens, embedder):
    rt_i = 0
    ct_i = 0
    last_end = 0
    output_tokens = []
    while rt_i < len(codebert_tokens) and ct_i < len(orig_tokens):
        if embedder.tokenizer.convert_tokens_to_string(codebert_tokens[:rt_i + 1]) == " ".join(orig_tokens[:ct_i + 1]):
            current_token_idxs = np.arange(last_end, rt_i + 1)
            output_tokens.append(current_token_idxs)
            last_end = rt_i + 1
            ct_i += 1
        rt_i += 1
    return output_tokens


def get_matched_labels_binary(code_token_id_mapping, query_token, nlp_cache):
    
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    matcher.add(query_token, [[{"TEXT": {"REGEX": query_token}}]])

    matches = [matcher(nlp_t) for nlp_t in nlp_cache]
    tags = [1 if match else 0 for i, match in enumerate(matches) for idx in code_token_id_mapping[i]]
    return np.asarray(tags)


def get_matched_labels_binary_v2(tokens, code_token_id_mapping, query_token):
    matches = [1 if re.search(query_token, t) else 0 for t in tokens]
    tags = np.zeros(max(code_token_id_mapping[-1]) + 1)
    for i, t in enumerate(tokens):
        if re.search(query_token, t):
            tags[code_token_id_mapping[i]] = 1
    return np.asarray(tags)


def get_static_labels_binary(code_token_id_mapping, query_token, static_tags):
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
    is_datatype = [label for label in static_labels if label in query_token ]
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
    datatype_regex_matches = {'dict': ['dict', 'map'], 
                          'list': ['list', 'arr'], 
                          'tuple': ['tuple'], 
                          'int': ['count', 'cnt', 'integer'], 
                          'file': ['file'], 
                          'enum': ['enum'], 
                          'string':['str', 'char', 'unicode', 'ascii'], 
                          'path': ['path', 'dir'], 
                          'bool': ['true', 'false', 'bool'],
                         }
    is_datatype = [datatype for datatype in datatype_regex_matches for regex_match in datatype_regex_matches[datatype] if regex_match in query_token]
    regex_match = np.zeros(max(code_token_id_mapping[-1]) + 1)
    for datatype in is_datatype:
        # because the data has been truncated, iterate over the truncated version
        for i in range(len(code_token_id_mapping)):
            tags = regex_tags[i]
            if datatype in tags:
                for idx in code_token_id_mapping[i]:
                    regex_match[idx] = 1
    return regex_match


def generate_finetuning_data(embedder, data):
    res_data = []
    res_scores = []

    for it in tqdm(range(len(data)), total=len(data), desc="Row: "):
        # sample some query and some code, half the cases will have the correct pair, the other half the cases will have an incorrect pair
        for pair in ['correct_pair', 'incorrect_pair']:
            if pair == 'correct_pair':
                doc = data['docstring_tokens'][it]
                code = data['alt_code_tokens'][it]
                static_tags = data['static_tags'][it]
                regex_tags = data['regex_tags'][it]
            else:
                # to make this reproducible
                np.random.seed(it)
                random_idx = np.random.choice(np.arange(len(data)), 1)[0]
                doc = data['docstring_tokens'][it]
                code = data['alt_code_tokens'][random_idx]
                static_tags = data['static_tags'][random_idx]
                regex_tags = data['regex_tags'][random_idx]
            if len(doc) == 0 or len(code) == 0:
                continue
            noun_tokens = extract_noun_tokens(' '.join(doc))
            # converting the docstring and code tokens into CodeBERT inputs
            # CodeBERT inputs are limited by 512 tokens, so this will truncate the inputs
            inputs = embedder.get_feature_inputs(' '.join(doc), ' '.join(code))
            separator = np.where(inputs['input_ids'][0].cpu().numpy() == embedder.tokenizer.sep_token_id)[0][0]
            # ignore CLS tokens at the beginning and at the end
            query_token_ids, code_token_ids = inputs['input_ids'][0][1:separator], inputs['input_ids'][0][separator + 1:-1]

            # get truncated version of code and query
            truncated_code_tokens = embedder.tokenizer.convert_ids_to_tokens(code_token_ids)
            truncated_query_tokens = embedder.tokenizer.convert_ids_to_tokens(query_token_ids)

            # mapping from CodeBERT tokenization to our dataset tokenization
            noun_token_id_mapping = np.asarray(get_word_to_roberta_tokens(doc, 
                                                                          truncated_query_tokens, 
                                                                          noun_tokens, embedder), dtype=object)

            code_token_id_mapping = np.asarray(get_code_to_roberta_tokens(code, 
                                                                          truncated_code_tokens, 
                                                                          embedder), dtype=object)
            if noun_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
                continue

            # get CodeBERT embedding for the example
            embedding = embedder.get_embeddings(inputs).cpu().detach().numpy()
            query_embedding, code_embedding = embedding[1:separator], embedding[separator + 1:-1]
            noun_token_embeddings = filter_embedding_by_id(query_embedding, noun_token_id_mapping)

            # extract positive pairs and sample negative pairs
            for nti, nte, nt in zip(noun_token_id_mapping, noun_token_embeddings, noun_tokens):
                # check for regex and static matches
                pos_sample_idxs = []
                idxs = get_matched_labels_binary_v2(code[:len(code_token_id_mapping)], code_token_id_mapping, nt).nonzero()[0]
                if idxs.size:
                    pos_sample_idxs.extend(idxs)
                idxs = get_static_labels_binary(code_token_id_mapping, nt, static_tags).nonzero()[0]
                if idxs.size:
                    pos_sample_idxs.extend(idxs)
                idxs = get_regex_labels_binary(code_token_id_mapping, nt, regex_tags).nonzero()[0]
                if idxs.size:
                    pos_sample_idxs.extend(idxs)
                # add positive example
                if len(pos_sample_idxs) > 0:
                    pos_sample_idxs = np.unique(pos_sample_idxs)
                    pos_samples_code_emb = code_embedding[pos_sample_idxs]
                    tiled_nte = np.tile(nte, (pos_samples_code_emb.shape[0], 1))
                    res_data.extend(np.hstack((tiled_nte, pos_samples_code_emb)))
                    res_scores.extend(np.ones(pos_samples_code_emb.shape[0]))

                # sample random number of negative examples
                num_neg_samples = np.sum(np.random.binomial(n=10, p=P))
                unique_ids, counts = np.unique(code[:len(code_token_id_mapping)], return_counts=True)
                id_freq_dict = {uid:c for uid, c in zip(unique_ids, counts)}
                p = np.asarray([1/id_freq_dict[i] for i in code[:len(code_token_id_mapping)]])
                p = p / np.sum(p)
                neg_sample_idxs = np.random.choice(np.arange(len(code_token_id_mapping)), num_neg_samples, replace=False, p=p)
                neg_sample_idxs = np.delete(neg_sample_idxs, np.where(neg_sample_idxs in pos_sample_idxs)[0])
                
                attempt = 0
                while neg_sample_idxs.size == 0 and attempt < 5:
                    attempt += 1
                    neg_sample_idxs = np.random.choice(np.arange(len(code_token_id_mapping)), num_neg_samples, replace=False, p=p)
                    neg_sample_idxs = np.delete(neg_sample_idxs, np.where(neg_sample_idxs in pos_sample_idxs)[0])
                if attempt == 5:
                    continue
                
                neg_sample_code_emb = filter_embedding_by_id(code_embedding, code_token_id_mapping[neg_sample_idxs])
                tiled_nte = np.tile(nte, (neg_sample_code_emb.shape[0], 1))
                res_data.extend(np.hstack((tiled_nte, neg_sample_code_emb)))
                res_scores.extend(np.zeros(len(neg_sample_code_emb)))
                if not np.all(np.asarray([len(r) for r in res_data[-1000:]]) == 1536):
                    raise Exception
    res_data = np.asarray(res_data)
    res_scores = np.asarray(res_scores)
    return res_data, res_scores


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if len(sys.argv) > 3:
        device = sys.argv[3]
    else:
        device = None
    data = pd.read_json(input_file, lines=True)
    embedder = Embedder(device)
    train_data, train_labels = generate_finetuning_data(embedder, data)
    np.save(f'{output_file}_data.npy', train_data)
    np.save(f'{output_file}_scores.npy', train_labels)

if __name__ == '__main__':
    main()
