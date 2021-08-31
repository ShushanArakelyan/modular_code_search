import argparse
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scoring.embedder import Embedder
from .utils import get_ground_truth_matches, get_noun_phrases

P = 0.7
VERSION="CLS"


def sample_random_code_tokens(code, code_token_id_mapping):
    num_samples = np.sum(np.random.binomial(n=20, p=P))
    unique_ids, counts = np.unique(code[:len(code_token_id_mapping)], return_counts=True)
    id_freq_dict = {uid: c for uid, c in zip(unique_ids, counts)}
    p = np.asarray([1 / id_freq_dict[i] for i in code[:len(code_token_id_mapping)]])
    p = p / np.sum(p)
    num_samples = min(num_samples, len(code_token_id_mapping))
    orig_tokens_sample_idxs = np.random.choice(np.arange(len(code_token_id_mapping)),
                                                   num_samples,
                                                   replace=False, p=p)
    sample_idxs = []
    for idx in orig_tokens_sample_idxs:
        sample_idxs.extend(code_token_id_mapping[idx])
    sample_idxs = np.asarray(sample_idxs)
    return sample_idxs


def cls_scoring(code, bceloss, scorer, embedder_out, pos_idxs_for_phrase):
    word_token_id_mapping, word_token_embeddings, code_token_id_mapping, \
    code_embedding, _, truncated_code_tokens, cls_token_embedding = embedder_out
    if len(pos_idxs_for_phrase) == 0:
        return None
    emb = cls_token_embedding.unsqueeze(0)

    # sample random number of negative examples
    neg_sample_idxs = []
    attempt = -1
    while neg_sample_idxs.size == 0 and attempt < 5:
        attempt += 1
        neg_sample_idxs = sample_random_code_tokens(code, code_token_id_mapping)
        neg_sample_idxs = neg_sample_idxs[~np.in1d(neg_sample_idxs, pos_idxs_for_phrase)]
    if attempt == 5:
        return None

    tiled_emb = emb.repeat(len(pos_idxs_for_phrase) + len(neg_sample_idxs), 1)
    ground_truth_scores =torch.cat((torch.FloatTensor(np.ones((len(pos_idxs_for_phrase), 1))),
                                   torch.FloatTensor(np.zeros((len(neg_sample_idxs), 1)))), dim=0).to(device)
    pos_samples_selected = torch.index_select(code_embedding,
                                              index=torch.LongTensor(pos_idxs_for_phrase).to(device), dim=0)
    neg_samples_selected = torch.index_select(code_embedding,
                                              index=torch.LongTensor(pos_idxs_for_phrase).to(device), dim=0)
    all_samples = torch.cat((pos_samples_selected, neg_samples_selected), dim=0)
    forward_input = torch.cat((tiled_emb, all_samples), dim=1)
    scorer_out = scorer.forward(forward_input)
    loss = bceloss(scorer_out, ground_truth_scores)
    return loss, len(pos_idxs_for_phrase) + len(neg_sample_idxs)


def run_epoch(data, scorer, embedder, op, bceloss, writer, writer_epoch, device, save_every, checkpoint_prefix):
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
                ccg_parse = data['ccg_parse'][it]
            else:
                np.random.seed(it)
                random_idx = np.random.choice(np.arange(len(data)), 1)[0]
                doc = data['docstring_tokens'][it]
                ccg_parse = data['ccg_parse'][it]
                code = data['alt_code_tokens'][random_idx]
                static_tags = data['static_tags'][random_idx]
                regex_tags = data['regex_tags'][random_idx]
            if len(doc) == 0 or len(code) == 0:
                continue
            phrases = get_noun_phrases(ccg_parse)
            for phrase in phrases:
                op.zero_grad()
                embedder_out = embedder.embed(phrase, code)
                if embedder_out is None:
                    continue
                word_token_id_mapping, word_token_embeddings, code_token_id_mapping, \
                code_embedding, _, truncated_code_tokens, cls_token_embedding = embedder_out
                if word_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
                    continue
                loss = None
                loss_normalization = 0
                # extract positive pairs and sample negative pairs
                pos_idxs_for_phrase = []
                for token in phrase:
                    pos_sample_idxs = np.unique(
                        get_ground_truth_matches(token, code, code_token_id_mapping, static_tags, regex_tags))
                    pos_idxs_for_phrase.extend(pos_sample_idxs)
                # version 1 - use CLS token embedding as phrase embedding
                # version 2 - average word_token_embeddings to get a phrase embedding
                # version 3 - tile word_token_embeddings and make pair-wise predictions
                if VERSION == 'CLS':
                    scoring_out = cls_scoring(bceloss, scorer, embedder_out, pos_idxs_for_phrase)
                    if scoring_out is None:
                        continue
                    loss_i, batch_size = scoring_out
                    if loss is None:
                        loss = loss_i
                    else:
                        loss += loss_i
                    loss_normalization += batch_size

                # for wti, wte, token in zip(word_token_id_mapping, word_token_embeddings, phrase):
                #     wte = wte.unsqueeze(0)
                #     # check for regex and static matches
                #     if len(pos_idxs_for_phrase) > 0:
                #         tiled_nte = wte.repeat(len(pos_idxs_for_phrase), 1)
                #         ground_truth_scores = torch.FloatTensor(np.ones((len(pos_idxs_for_phrase), 1))).to(device)
                #         pos_samples_selected = torch.index_select(code_embedding,
                #                                                   index=torch.LongTensor(pos_idxs_for_phrase).to(device), dim=0)
                #         forward_input = torch.cat((tiled_nte, pos_samples_selected), dim=1)
                #         scorer_out = scorer.forward(forward_input)
                #         if loss is None:
                #             loss = bceloss(scorer_out, ground_truth_scores)
                #         else:
                #             loss += bceloss(scorer_out, ground_truth_scores)
                #     loss_normalization += len(pos_idxs_for_phrase)
                #     # sample random number of negative examples
                #     num_neg_samples = np.sum(np.random.binomial(n=20, p=P))
                #     unique_ids, counts = np.unique(code[:len(code_token_id_mapping)], return_counts=True)
                #     id_freq_dict = {uid: c for uid, c in zip(unique_ids, counts)}
                #     p = np.asarray([1 / id_freq_dict[i] for i in code[:len(code_token_id_mapping)]])
                #     p = p / np.sum(p)
                #     num_neg_samples = min(num_neg_samples, len(code_token_id_mapping))
                #     orig_tokens_neg_sample_idxs = np.random.choice(np.arange(len(code_token_id_mapping)),
                #                                                    num_neg_samples,
                #                                                    replace=False, p=p)
                #     neg_sample_idxs = []
                #     for idx in orig_tokens_neg_sample_idxs:
                #         neg_sample_idxs.extend(code_token_id_mapping[idx])
                #     neg_sample_idxs = np.asarray(neg_sample_idxs)
                #     neg_sample_idxs = neg_sample_idxs[~np.in1d(neg_sample_idxs, pos_sample_idxs)]
                #
                #     attempt = 0
                #     while neg_sample_idxs.size == 0 and attempt < 5:
                #         attempt += 1
                #         orig_tokens_neg_sample_idxs = np.random.choice(
                #             np.arange(len(code_token_id_mapping)), num_neg_samples,
                #             replace=False, p=p)
                #         neg_sample_idxs = []
                #         for idx in orig_tokens_neg_sample_idxs:
                #             neg_sample_idxs.extend(code_token_id_mapping[idx])
                #         neg_sample_idxs = np.asarray(neg_sample_idxs)
                #         neg_sample_idxs = neg_sample_idxs[~np.in1d(neg_sample_idxs, pos_sample_idxs)]
                #     if attempt == 5:
                #         continue
                #
                #     tiled_nte = wte.repeat(len(neg_sample_idxs), 1)
                #     ground_truth_scores = torch.FloatTensor(np.zeros((len(neg_sample_idxs), 1))).to(device)
                #     neg_samples_selected = torch.index_select(code_embedding,
                #                                               index=torch.LongTensor(neg_sample_idxs).to(device), dim=0)
                #     forward_input = torch.cat((tiled_nte, neg_samples_selected), dim=1)
                #     scorer_out = scorer.forward(forward_input)
                #     if loss is None:
                #         loss = bceloss(scorer_out, ground_truth_scores)
                #     else:
                #         loss += bceloss(scorer_out, ground_truth_scores)
                #     loss_normalization += len(neg_sample_idxs)
                loss /= loss_normalization
                loss.backward()
                op.step()
                cumulative_loss.append(loss.item())
        if it > 0 and it % 100 == 0:
            writer_epoch += 1
            writer.add_scalar("Loss/train", np.mean(cumulative_loss[-100:]), writer_epoch)
        if it > 0 and it % save_every == 0:
            torch.save({"scorer": scorer.state_dict(),
                        "embedder": embedder.model.state_dict(),
                        "optimizer": op.state_dict()}, checkpoint_prefix + f'{it}.tar')
    return cumulative_loss, writer_epoch


def main():
    parser = argparse.ArgumentParser(description='Fine-tune codebert model for attend module')
    parser.add_argument('--load_from_checkpoint', dest='checkpoint', type=str,
                        help='continue training from checkpoint')
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='training data directory', required=True)
    parser.add_argument('--scorer_only', default=False, action='store_true')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        help='number of epochs to train')

    args = parser.parse_args()

    device = args.device

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    writer = SummaryWriter(f'/home/shushan/modular_code_search/runs/{dt_string}')
    print("Writing to tensorboard: ", dt_string)

    if args.scorer_only:
        embedder = Embedder(device, model_eval=True)
        scorer = torch.nn.Sequential(torch.nn.Linear(embedder.get_dim() * 2, embedder.get_dim()),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(embedder.get_dim(), 1)).to(device)

        op = torch.optim.Adam(list(scorer.parameters()), lr=1e-8)
    else:
        embedder = Embedder(device, model_eval=False)
        scorer = torch.nn.Sequential(torch.nn.Linear(embedder.get_dim() * 2, embedder.get_dim()),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(embedder.get_dim(), 1)).to(device)
        op = torch.optim.Adam(list(scorer.parameters()) + list(embedder.model.parameters()), lr=1e-8)
    bceloss = torch.nn.BCEWithLogitsLoss(reduction='sum')

    if args.checkpoint:
        models = torch.load(args.checkpoint, map_location=device)
        checkpoint_dir = '/'.join(args.checkpoint.split('/')[:-1])
        file_name = args.checkpoint.split('/')[-1]
        epoch_to_start = int(file_name.split('_')[1])
        datafile_to_start = int(file_name.split('_')[-1].split('.')[0]) + 1
        print(f"Continuing from epoch: {epoch_to_start}, datafile: {datafile_to_start}")
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
        epoch_to_start = 0
        datafile_to_start = -1
        checkpoint_dir = f'/home/shushan/finetuned_scoring_models/{dt_string}'
        print("Checkpoints will be saved in ", checkpoint_dir)

        import os
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    if args.num_epochs:
        num_epochs = args.num_epochs
    else:
        num_epochs = 10

    train_writer_epoch = 0

    train_files = []
    for file in glob.glob(args.data_dir + '/*'):
        if not file.endswith('jsonl.gz') and not file.endswith('jsonl'):
            print("skipping: ", file)
        else:
            train_files.append(file)
    train_files = natsorted(train_files)

    print('This run will not be performing validation while training, number of epochs to run: ', num_epochs)
    for epoch in range(epoch_to_start, num_epochs):
        for i, input_file_name in enumerate(train_files):
            if i < datafile_to_start:
                continue
            else:
                datafile_to_start = -1
            print("Processing file: ", input_file_name)
            data = pd.read_json(input_file_name, lines=True)
            total_loss, train_writer_epoch = run_epoch(data, scorer, embedder, op, bceloss, writer, train_writer_epoch,
                                                       device, save_every=5000,
                                                       checkpoint_prefix=checkpoint_dir + f'/model_{epoch}_ep_{i}')
        datafile_to_start = -1


if __name__ == '__main__':
    main()
