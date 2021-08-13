import numpy as np
import pandas as pd
import spacy
import sys
import torch

from tqdm import tqdm

from utils.tokenize_and_label import new_query_matches
from .embedder import Embedder


nlp = spacy.load("en_core_web_md")


def sub_sample(matches, embeddings):
    """Having all the scores, the function samples half positive, half negative examples."""
    match_num = int(np.sum(matches))
    scores = np.zeros(match_num * 2)
    embeds = np.zeros((match_num * 2, 768))

    ind0 = np.random.choice(np.where(matches == 0)[0], match_num, replace=False)
    ind1 = np.where(matches == 1)[0]
    index = np.concatenate((ind0, ind1))

    for i in range(len(embeddings)):
        for k, j in enumerate(index):
            embeds[k] = embeddings[j]
            scores[k] = 1 if j in ind1 else 0
    return scores, np.asarray(embeds)


def filter_embeddings(embedder, code_tokens, code_embeddings, query_token, nlp_cache, static_tags):
    """Gets the scores for the given query-code tokens and samples negative and positive examples for training."""
    with torch.no_grad():
        regex_match, static_match = new_query_matches(query_token, code_tokens, nlp_cache, static_tags,
                                                      embedder.tokenizer)
        match = regex_match + static_match

        # for instances where there is both regex and static label
        match[match > 1] = 1

        # for instances when there was no match between code tokens and query token
        if sum(match) == 0:
            return None, None

        # gets the embedding of our desired query token
        return sub_sample(match, code_embeddings.cpu().detach())


def extract_noun_tokens(doc):
    """Having the docstring, the function returns only the word that are nouns."""
    tokens = nlp(doc)
    pos_tags = [token.tag_ for token in tokens]
    # checks whether the word has expected pos tag
    noun_tokens = []
    for i, pos in enumerate(pos_tags):
        if pos.startswith('N'):
            # we lower the word, as many words are recognized by tokenizer when they are lowered
            noun_tokens.append(tokens[i].text.lower())
    return noun_tokens


def get_static_tag(code_tokens, static_tags):
    pairs = []
    static_idx = [i for i, tag in enumerate(static_tags) if len(tag) > 0] + [0]

    current = static_idx[0]
    sub_pair = []

    for i in range(1, len(static_idx)):
        sub_pair.append(current)
        if current + 1 != static_idx[i]:
            pairs.append(sub_pair)
            sub_pair = []
        current = static_idx[i]

    static_pairs = []
    for pair in pairs:
        if len(pair) == 1:
            code = [code_tokens[pair[0]]]
        else:
            code = [code_tokens[i] for i in pair]
        static_pairs.append([code, static_tags[pair[0]]])

    return static_pairs


def process_file_data(embedder, data, file_name):
    """Given the data, it gets the docstring and code tokens, preprocesses it, combines the corresponding query-code
    embeddings together and saves them into numpy file."""
    embed_data = []
    embed_scores = []

    f = open(f'{file_name}_exception.txt', 'w')
    for i, row in tqdm(data.iterrows(), total=len(data), desc="Row: "):
        try:
            doc = row['docstring_tokens']
            code = row['alt_code_tokens']
            static_tags = row['static_tags']

            noun_tokens = extract_noun_tokens(' '.join(doc))
            static_tags = get_static_tag(code, static_tags)

            doc = ' '.join(noun_tokens)
            code = ' '.join(code).lower()

            # converting the docstring and code tokens into CodeBERT inputs
            # code = [InputExample(0, code, label="0")]
            # doc = [InputExample(0, doc, label="0")]
            qinputs = embedder.get_feature_inputs(doc)
            cinputs = embedder.get_feature_inputs(code)

            code_tokens = embedder.tokenizer.convert_ids_to_tokens(cinputs['input_ids'][0])

            # caching for faster processing
            nlp_cache = []
            for t in code_tokens:
                nlp_cache.append(nlp(t))

            code_embeddings = embedder.get_embeddings(cinputs)

            for query_token in noun_tokens:
                query_embedding = embedder.get_token_embedding(qinputs, embedder.get_embeddings(qinputs),
                                                               query_token)
                scores, sampled_code_embeddings = filter_embeddings(embedder, code_tokens, code_embeddings, query_token,
                                                                    nlp_cache, static_tags)
                # when there was no match between the query token and code tokens
                if sampled_code_embeddings is None:
                    continue
                for j in range(len(sampled_code_embeddings)):
                    embed_data.append(np.concatenate((query_embedding.cpu().detach(), sampled_code_embeddings[j])))
                    embed_scores.append(scores[j])

        except Exception as e:
            f.write(f'{e} \n')
            continue
    f.close()
    return np.stack(embed_data), np.asarray(embed_scores)


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if len(sys.argv) > 3:
        device = sys.argv[3]
    else:
        device = 'cpu'
    data = pd.read_json(input_file, lines=True)
    embedder = Embedder(device)
    train_data, train_label = process_file_data(embedder, data, output_file)
    np.save(f'{output_file}_data.npy', train_data)
    np.save(f'{output_file}_scores.npy', train_label)

if __name__ == '__main__':
    main()
