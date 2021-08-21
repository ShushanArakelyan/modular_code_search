import ast
import asttokens
import argparse
import glob
import io
import numpy as np
import pandas as pd
import spacy
import sys
import torch
import time
import re


from datetime import datetime
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scoring.embedder import Embedder

device = 'cuda:4'
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
writer = SummaryWriter(f'/home/shushan/modular_code_search/runs/{dt_string}')
print("Writing to tensorboard: ", dt_string)

nlp = spacy.load("en_core_web_md")
dim = 768
P = 0.7

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
        nte = torch.index_select(query_embedding, index=torch.LongTensor(nti.astype(int)).to(device), dim=0)
        noun_token_embeddings.append(torch.unsqueeze(torch.mean(nte, dim=0), 0))
    noun_token_embeddings = torch.cat(noun_token_embeddings)
    return noun_token_embeddings


def get_word_to_roberta_tokens(orig_tokens, codebert_tokens, noun_tokens, embedder):
    rt_i = 0
    nt_i = 0
    last_end = 0
    output_tokens = []
    while rt_i < len(codebert_tokens) and nt_i < len(orig_tokens):
        if embedder.tokenizer.convert_tokens_to_string(codebert_tokens[:rt_i + 1]).lower() == " ".join(orig_tokens[:nt_i + 1]).lower():
            current_token_idxs = np.arange(last_end, rt_i + 1)
            if orig_tokens[nt_i].lower() in noun_tokens:
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


def run_epoch(data, scorer, embedder, op, bceloss, writer_epoch, train=True):
    cumulative_loss = []
    for it in tqdm(range(len(data)), total=len(data), desc="Row: "):
        # sample some query and some code, half the cases will have the correct pair, 
        # the other half the cases will have an incorrect pair
        for pair in ['correct_pair', 'incorrect_pair']:
            if pair == 'correct_pair':
                doc = data['docstring_tokens'][it]
                code = data['alt_code_tokens'][it]
                static_tags = data['static_tags'][it]
                regex_tags = data['regex_tags'][it]
            else:
                np.random.seed(it)
                random_idx = np.random.choice(np.arange(len(data)), 1)[0]
                doc = data['docstring_tokens'][it]
                code = data['alt_code_tokens'][random_idx]
                static_tags = data['static_tags'][random_idx]
                regex_tags = data['regex_tags'][random_idx]
            if len(doc) == 0 or len(code) == 0:
                continue
            op.zero_grad()
            noun_tokens = extract_noun_tokens(' '.join(doc))
            # converting the docstring and code tokens into CodeBERT inputs
            # CodeBERT inputs are limited by 512 tokens, so this will truncate the inputs
            inputs = embedder.get_feature_inputs(' '.join(doc), ' '.join(code))
            separator = np.where(
                inputs['input_ids'][0].cpu().numpy() == embedder.tokenizer.sep_token_id)[0][0]
            # ignore CLS tokens at the beginning and at the end
            query_token_ids = inputs['input_ids'][0][1:separator]
            code_token_ids = inputs['input_ids'][0][separator + 1:-1]

            # get truncated version of code and query
            truncated_code_tokens = embedder.tokenizer.convert_ids_to_tokens(code_token_ids)
            truncated_query_tokens = embedder.tokenizer.convert_ids_to_tokens(query_token_ids)

            # mapping from CodeBERT tokenization to our dataset tokenization
            noun_token_id_mapping = np.asarray(get_word_to_roberta_tokens(doc, 
                                                                          truncated_query_tokens, 
                                                                          noun_tokens, embedder),
                                               dtype=object)

            code_token_id_mapping = np.asarray(get_code_to_roberta_tokens(code, 
                                                                          truncated_code_tokens, 
                                                                          embedder), 
                                               dtype=object)
            if noun_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
                continue

            # get CodeBERT embedding for the example
            embedding = embedder.get_embeddings(inputs)
            query_embedding, code_embedding = embedding[1:separator], embedding[separator + 1:-1]
            noun_token_embeddings = filter_embedding_by_id(query_embedding, noun_token_id_mapping)

            loss = None
            # extract positive pairs and sample negative pairs
            for nti, nte, nt in zip(noun_token_id_mapping, noun_token_embeddings, noun_tokens):
                nte = nte.unsqueeze(0)
                # check for regex and static matches
                pos_sample_idxs = []
                idxs = get_matched_labels_binary_v2(code[:len(code_token_id_mapping)], 
                                                    code_token_id_mapping, nt).nonzero()[0]
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
#                     for pos_sample_idx in pos_sample_idxs:
#                         pos_sample_selected = torch.index_select(
#                             code_embedding,index=torch.LongTensor(
#                                 [pos_sample_idx]).to(device), dim=0)
#                         forward_input = torch.cat((nte, pos_sample_selected), dim=1)
#                         scorer_out = scorer.forward(forward_input)
#                         if not loss:
#                             loss = bceloss(scorer_out, torch.FloatTensor([[1]]).to(device))
#                         else:
#                             loss += bceloss(scorer_out, torch.FloatTensor([[1]]).to(device))
                    ###############
                    tiled_nte = nte.repeat(len(pos_sample_idxs), 1)
                    ground_truth_scores = torch.FloatTensor(np.ones((len(pos_sample_idxs), 1))).to(device)
                    pos_samples_selected = torch.index_select(code_embedding, index=torch.LongTensor(pos_sample_idxs).to(device), dim=0)
                    forward_input = torch.cat((tiled_nte, pos_samples_selected), dim=1)
                    scorer_out = scorer.forward(forward_input)
                    if loss is None:
                        loss = bceloss(scorer_out, ground_truth_scores)
                    else:
                        loss += bceloss(scorer_out, ground_truth_scores)
                    ###############

                # sample random number of negative examples
                num_neg_samples = np.sum(np.random.binomial(n=20, p=P))
                unique_ids, counts = np.unique(code[:len(code_token_id_mapping)], return_counts=True)
                id_freq_dict = {uid:c for uid, c in zip(unique_ids, counts)}
                p = np.asarray([1/id_freq_dict[i] for i in code[:len(code_token_id_mapping)]])
                p = p / np.sum(p)
                neg_sample_idxs = np.random.choice(np.arange(len(code_token_id_mapping)), 
                                                   num_neg_samples, 
                                                   replace=False, p=p)
                neg_sample_idxs = neg_sample_idxs[~np.in1d(neg_sample_idxs, pos_sample_idxs)]
                
                attempt = 0
                while neg_sample_idxs.size == 0 and attempt < 5:
                    attempt += 1
                    neg_sample_idxs = np.random.choice(
                        np.arange(len(code_token_id_mapping)), num_neg_samples, 
                                                       replace=False, p=p)
                    neg_sample_idxs = neg_sample_idxs[~np.in1d(neg_sample_idxs, pos_sample_idxs)]
                if attempt == 5:
                    continue
                    
                ###############    
                tiled_nte = nte.repeat(len(neg_sample_idxs), 1)
                ground_truth_scores = torch.FloatTensor(np.zeros((len(neg_sample_idxs), 1))).to(device)
                neg_samples_selected = torch.index_select(code_embedding, index=torch.LongTensor(neg_sample_idxs).to(device), dim=0)
                forward_input = torch.cat((tiled_nte, neg_samples_selected), dim=1)
                scorer_out = scorer.forward(forward_input)
                if loss is None:
                    loss = bceloss(scorer_out, ground_truth_scores)
                else:
                    loss += bceloss(scorer_out, ground_truth_scores)
                ###############                    

#                 for neg_sample_idx in neg_sample_idxs:
#                     neg_sample_selected = torch.index_select(
#                         code_embedding, index=torch.LongTensor([neg_sample_idx]).to(device), 
#                                                              dim=0)
#                     forward_input = torch.cat((nte, neg_sample_selected), dim=1)
#                     scorer_out = scorer.forward(forward_input)
#                     if not loss:
#                         loss = bceloss(scorer_out, torch.FloatTensor([[0]]).to(device))
#                     else:
#                         loss += bceloss(scorer_out, torch.FloatTensor([[0]]).to(device))
                loss /= (len(pos_sample_idxs) + len(neg_sample_idxs))
            if train:
                loss.backward()
                op.step()
            cumulative_loss.append(loss.item())
        if it % 100 == 0:
            writer_epoch += 1
            writer.add_scalar("Loss/train" if train else "Loss/valid", 
                              np.mean(cumulative_loss[-100:]), writer_epoch)
    return cumulative_loss, writer_epoch

    
def main():
    parser = argparse.ArgumentParser(description='Fine-tune codebert model for attend module')
    parser.add_argument('--load_from_checkpoint', dest='checkpoint', type=str,
                        help='continue training from checkpoint')
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='training data directory', required=True)
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='validation data directory', required=True)
    args = parser.parse_args()
    if args.device:
        global device
        device = args.device
        print("Device: ", device)
    else:
        device = 'cuda:0'
    embedder = Embedder(device)
    scorer = torch.nn.Sequential(torch.nn.Linear(dim*2, dim),
                       torch.nn.ReLU(),
                       torch.nn.Linear(dim, 1)).to(device)
    op = torch.optim.Adam(list(scorer.parameters()) + list(embedder.model.parameters()), lr=1e-8)
    bceloss = torch.nn.BCEWithLogitsLoss(reduction='sum')

    if args.checkpoint:
        models = torch.load(args.checkpoint)
        checkpoint_dir = '/'.join(args.checkpoint.split('/')[:-1])
        scorer.load_state_dict(models['scorer'])
        print("Scorer device: ", next(scorer.parameters()).device)
        scorer = scorer.to(device)
        print("Scorer device: ", next(scorer.parameters()).device)
        embedder.model.load_state_dict(models['embedder'])
        print("Embedder device: ", next(embedder.model.parameters()).device)
        embedder.model = embedder.model.to(device)
        print("Embedder device: ", next(embedder.model.parameters()).device)
        op.load_state_dict(models['optimizer'])
    else:
        checkpoint_dir = f'/home/shushan/finetuned_scoring_models/{dt_string}'
        print("Checkpoints will be saved in ", checkpoint_dir)
        
        import os
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
    train_writer_epoch = 0
    valid_writer_epoch = 0
    
    train_files = []
    for file in glob.glob(args.data_dir + '/*'):
        if not file.endswith('jsonl.gz') and not file.endswith('jsonl'):
            print("skipping: ", file)
        else:
            train_files.append(file)
    train_files = natsorted(train_files)

#     valid_data = pd.read_json(args.valid_file_name, lines=True)
    valid_data = None
    print('This run will not be performing validation while training')
    for epoch in range(5):
        for i, input_file_name in enumerate(train_files):
            print("Processing file: ", input_file_name)
            data = pd.read_json(input_file_name, lines=True)
            total_loss, train_writer_epoch = run_epoch(data, scorer, embedder, 
                                                       op, bceloss, train_writer_epoch, train=True)
            torch.save({"scorer": scorer.state_dict(), 
            "embedder": embedder.model.state_dict(), 
            "optimizer": op.state_dict()}, checkpoint_dir + f'/model_{epoch}_ep_{i}.tar')
#             valid_loss, valid_writer_epoch = run_epoch(valid_data, scorer, embedder, 
#                                                        op, bceloss, valid_writer_epoch, train=False)
            
if __name__ == '__main__':
    main() 