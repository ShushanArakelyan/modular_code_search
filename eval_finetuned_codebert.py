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
from finetune_codebert import extract_noun_tokens, get_word_to_roberta_tokens, get_code_to_roberta_tokens, filter_embedding_by_id, get_matched_labels_binary_v2, get_static_labels_binary, get_regex_labels_binary

device = 'cuda:4'
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
writer = SummaryWriter(f'/home/shushan/modular_code_search/runs/{dt_string}')
print("Writing to tensorboard: ", dt_string)

nlp = spacy.load("en_core_web_md")
dim = 768


def run_eval_epoch(data, scorer, embedder, split_point=0.5):
    f1_scores = []
    exact_matches = []
    for it in range(len(data)):
        f1_scores_for_sample = []
        exact_matches_for_sample = []
        
        doc = data['docstring_tokens'][it]
        code = data['alt_code_tokens'][it]
        static_tags = data['static_tags'][it]
        regex_tags = data['regex_tags'][it]
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
        # get CodeBERT embedding for the example
        embedding = embedder.get_embeddings(inputs)
        if noun_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
                continue
        query_embedding, code_embedding = embedding[1:separator], embedding[separator + 1:-1]
        noun_token_embeddings = filter_embedding_by_id(query_embedding, noun_token_id_mapping)

        for nti, nte, nt in zip(noun_token_id_mapping, noun_token_embeddings, noun_tokens):
            nte = nte.unsqueeze(0)
            # check for regex and static matches
            ground_truth_idxs = []
            idxs = get_matched_labels_binary_v2(code[:len(code_token_id_mapping)], 
                                                code_token_id_mapping, nt).nonzero()[0]
            if idxs.size:
                ground_truth_idxs.extend(idxs)
            idxs = get_static_labels_binary(code_token_id_mapping, nt, static_tags).nonzero()[0]
            if idxs.size:
                ground_truth_idxs.extend(idxs)
            idxs = get_regex_labels_binary(code_token_id_mapping, nt, regex_tags).nonzero()[0]
            if idxs.size:
                ground_truth_idxs.extend(idxs)

            all_idxs = np.arange(len(truncated_code_tokens))
            predicted_idxs = []
            tiled_nte = nte.repeat(len(truncated_code_tokens), 1)
            forward_input = torch.cat((tiled_nte, code_embedding), dim=1)
            scorer_out = torch.sigmoid(scorer.forward(forward_input))
            scorer_out = scorer_out.squeeze().cpu().detach().numpy()
            predicted_idxs = np.where(scorer_out >= split_point)[0]
            S_g = len(ground_truth_idxs)
            S_a = len(predicted_idxs)
            intersection = len(np.intersect1d(predicted_idxs, ground_truth_idxs))
            exact_matches_for_sample.append(intersection)
            if S_g == 0:
                f1_scores_for_sample.append(1 if S_a == 0 else 0)
                continue
            if S_a == 0:
                P_t = 0
            else:
                P_t = intersection/S_a
            R_t = intersection/S_g
            if P_t == 0 and R_t == 0:
                f1_scores_for_sample.append(0)
            else:
                f1_scores_for_sample.append((2 * P_t * R_t)/(P_t + R_t))
        f1_scores.append(f1_scores_for_sample)
        exact_matches.append(exact_matches_for_sample)
    return f1_scores, exact_matches            
   
    
def find_split_point(data, scorer, embedder):
    avg_f1 = []
    split_points = np.arange(0, 1, 0.05)
    for i in split_points:
        f1_scores, exact_matches = run_eval_epoch(data, scorer, embedder, split_point=i)
        avg_f1.append(np.mean([np.mean(f1) for f1 in f1_scores]))
    return split_points, avg_f1
            
    
def main():
    parser = argparse.ArgumentParser(description='Fine-tune codebert model for attend module')
    parser.add_argument('--load_from_checkpoint', dest='checkpoint', type=str,
                        help='continue training from checkpoint', requried=True)
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on')
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='validation data directory', required=True)
    parser.add_argument('--train_file_name', dest='train_file_name', type=str,
                        help='data to find the best split point', required=True)
    args = parser.parse_args()
    if args.device:
        global device
        device = args.device
        print("Device: ", device)
    embedder = Embedder(device)
    scorer = torch.nn.Sequential(torch.nn.Linear(dim*2, dim),
                       torch.nn.ReLU(),
                       torch.nn.Linear(dim, 1)).to(device)
    models = torch.load(args.checkpoint)
    checkpoint_dir = '/'.join(args.checkpoint.split('/')[:-1])
    scorer.load_state_dict(models['scorer'])
    scorer = scorer.to(device)
    embedder.model.load_state_dict(models['embedder'])
    embedder.model = embedder.model.to(device)  
    
    train_data = pd.read_json(args.train_file_name, lines=True)
    split_points, f1_scores = find_split_point(train_data[:50], scorer, embedder)
    split_point = split_points[np.argmax(f1_scores)]
        
    valid_writer_epoch = 0
    
    valid_data = pd.read_json(args.valid_file_name, lines=True)
    for i, input_file_name in enumerate(train_files):
        valid_loss, valid_writer_epoch = run_eval_epoch(valid_data, scorer, embedder, 
                                                        op, bceloss, valid_writer_epoch, split_point=split_point)
            
if __name__ == '__main__':
    main() 