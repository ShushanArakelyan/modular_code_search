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
from .utils import get_ground_truth_matches, get_noun_phrases, embed_pair

P = 0.7
VERSION = "CLS"
# VERSION = "MEAN"
INCLUDE_MISMATCHED_PAIR = False
EMBED_SEPARATELY = False
DOWNSAMPLE_GT = False


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


def downsample_ground_truth(sampled_idxs, code, code_token_id_mapping, ds_prob=0.5):
    rev_code_token_id_mapping = {vi: k for k, v in enumerate(code_token_id_mapping) for vi in v}
    new_sampled_idxs = []
    n = len(rev_code_token_id_mapping)
    for idx in sampled_idxs:
        orig_token_neighborhood = [rev_code_token_id_mapping[i] for i in range(max(0, idx - 3), min(idx + 3, n))]
        neighborhood = [code[j] for j in np.unique(orig_token_neighborhood)]
        if 'def' in neighborhood or 'return' in neighborhood:
            if np.random.rand(1) > ds_prob:
                new_sampled_idxs.append(idx)
            else:
                continue
        else:
            new_sampled_idxs.append(idx)
    return new_sampled_idxs


def mean_scoring(code, bceloss, scorer, embedder_out, pos_idxs_for_phrase, device):
    word_token_id_mapping, word_token_embeddings, code_token_id_mapping, \
    code_embedding, _, truncated_code_tokens, cls_token_embedding = embedder_out
    if len(pos_idxs_for_phrase) == 0:
        return None

    # sample random number of negative examples
    neg_sample_idxs = np.zeros(0)
    attempt = -1
    while neg_sample_idxs.size == 0 and attempt < 5:
        attempt += 1
        neg_sample_idxs = sample_random_code_tokens(code, code_token_id_mapping)
        neg_sample_idxs = neg_sample_idxs[~np.in1d(neg_sample_idxs, pos_idxs_for_phrase)]
    if attempt == 5:
        return None
    emb = torch.mean(word_token_embeddings, dim=0, keepdim=True)

    tiled_emb = emb.repeat(len(pos_idxs_for_phrase) + len(neg_sample_idxs), 1)
    ground_truth_scores = torch.cat((torch.FloatTensor(np.ones((len(pos_idxs_for_phrase), 1))),
                                     torch.FloatTensor(np.zeros((len(neg_sample_idxs), 1)))), dim=0).to(device)
    pos_samples_selected = torch.index_select(code_embedding,
                                              index=torch.LongTensor(pos_idxs_for_phrase).to(device), dim=0)
    neg_samples_selected = torch.index_select(code_embedding,
                                              index=torch.LongTensor(neg_sample_idxs).to(device), dim=0)
    all_samples = torch.cat((pos_samples_selected, neg_samples_selected), dim=0)
    forward_input = torch.cat((tiled_emb, all_samples), dim=1)
    scorer_out = scorer.forward(forward_input)
    loss = bceloss(scorer_out, ground_truth_scores)
    return loss, len(pos_idxs_for_phrase) + len(neg_sample_idxs)


def cls_scoring(code, bceloss, scorer, embedder_out, pos_idxs_for_phrase, device):
    word_token_id_mapping, word_token_embeddings, code_token_id_mapping, \
    code_embedding, _, truncated_code_tokens, cls_token_embedding = embedder_out
    if len(pos_idxs_for_phrase) == 0:
        return None

    # sample random number of negative examples
    neg_sample_idxs = np.zeros(0)
    attempt = -1
    while neg_sample_idxs.size == 0 and attempt < 5:
        attempt += 1
        neg_sample_idxs = sample_random_code_tokens(code, code_token_id_mapping)
        neg_sample_idxs = neg_sample_idxs[~np.in1d(neg_sample_idxs, pos_idxs_for_phrase)]
    if attempt == 5:
        return None
    tiled_emb = cls_token_embedding.repeat(len(pos_idxs_for_phrase) + len(neg_sample_idxs), 1)
    ground_truth_scores = torch.cat((torch.FloatTensor(np.ones((len(pos_idxs_for_phrase), 1))),
                                     torch.FloatTensor(np.zeros((len(neg_sample_idxs), 1)))), dim=0).to(device)
    pos_samples_selected = torch.index_select(code_embedding,
                                              index=torch.LongTensor(pos_idxs_for_phrase).to(device), dim=0)
    neg_samples_selected = torch.index_select(code_embedding,
                                              index=torch.LongTensor(neg_sample_idxs).to(device), dim=0)
    all_samples = torch.cat((pos_samples_selected, neg_samples_selected), dim=0)
    forward_input = torch.cat((tiled_emb, all_samples), dim=1)
    scorer_out = scorer.forward(forward_input)
    loss = bceloss(scorer_out, ground_truth_scores)
    return loss, len(pos_idxs_for_phrase) + len(neg_sample_idxs)



def train_one_example(sample, scorer, embedder, op, bceloss, device):
    doc, code, static_tags, regex_tags, ccg_parse = sample
    if '\\' in ccg_parse:
        # TODO: this example is not parsed properly, skip for now, but handle somehow in the future
        return None
    phrases = get_noun_phrases(ccg_parse)
    cumulative_loss = 0
    for phrase in phrases:
        op.zero_grad()
        embedder_out = embed_pair(embedder, phrase, code, EMBED_SEPARATELY)
        if embedder_out is None:
            continue
        word_token_id_mapping, word_token_embeddings, code_token_id_mapping, \
        code_embedding, _, truncated_code_tokens, cls_token_embedding = embedder_out
        loss = None
        loss_normalization = 0
        # extract positive pairs and sample negative pairs
        pos_idxs_for_phrase = []
        for token in phrase:
            pos_sample_idxs = get_ground_truth_matches(token, code, code_token_id_mapping, static_tags, regex_tags)
            pos_idxs_for_phrase.extend(pos_sample_idxs)
        pos_idxs_for_phrase = np.unique(pos_idxs_for_phrase)
        if DOWNSAMPLE_GT:
            downsample_ground_truth(pos_idxs_for_phrase, code, code_token_id_mapping)
        # version 1 - use CLS token embedding as phrase embedding
        # version 2 - average word_token_embeddings to get a phrase embedding
        # version 3 - tile word_token_embeddings and make pair-wise predictions
        if VERSION == 'CLS':
            scoring_out = cls_scoring(code, bceloss, scorer, embedder_out, pos_idxs_for_phrase, device)
            if scoring_out is None:
                continue
            loss_i, batch_size = scoring_out
            if loss is None:
                loss = loss_i
            else:
                loss += loss_i
            loss_normalization += batch_size
        elif VERSION == "MEAN":
            scoring_out = cls_scoring(code, bceloss, scorer, embedder_out, pos_idxs_for_phrase, device)
            if scoring_out is None:
                continue
            loss_i, batch_size = scoring_out
            if loss is None:
                loss = loss_i
            else:
                loss += loss_i
            loss_normalization += batch_size
        loss /= loss_normalization
        loss.backward()
        op.step()
        cumulative_loss += loss.item()
    return cumulative_loss


def run_epoch(data, scorer, embedder, op, bceloss, writer, writer_epoch, device, checkpoint_prefix, save_every=None):
    cumulative_loss = []
    for it in tqdm(range(len(data)), total=len(data), desc="Row: "):
        # sample some query and some code, half the cases will have the correct pair, 
        # the other half the cases will have an incorrect pair
        if INCLUDE_MISMATCHED_PAIR:
            pairs = ['correct_pair', 'incorrect_pair']
        else:
            pairs = ['correct_pair']
        for pair in pairs:
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
            sample = (doc, code, static_tags, regex_tags, ccg_parse)
            out = train_one_example(sample, scorer, embedder, op, bceloss, device)
            if out is not None:
                cumulative_loss.append(out)
        if it > 0 and it % 100 == 0:
            writer_epoch += 1
            writer.add_scalar("Loss/train", np.mean(cumulative_loss[-100:]), writer_epoch)
        if save_every:
            if it > 0 and (it + 1) % save_every == 0:
                torch.save({"scorer": scorer.state_dict(),
                            "embedder": embedder.model.state_dict(),
                            "optimizer": op.state_dict()}, checkpoint_prefix + f'{it + 1}.tar')
    torch.save({"scorer": scorer.state_dict(),
                "embedder": embedder.model.state_dict(),
                "optimizer": op.state_dict()}, checkpoint_prefix + '.tar')
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
    parser.add_argument('--include_mismatched_pair', default=False, action='store_true')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        help='number of epochs to train')
    parser.add_argument('--embed_separately', dest='embed_separately', default=False, action='store_true',
                        help='Whether to embed the query and code in a single instance or separate instances')
    parser.add_argument('--downsample_gt', dest='downsample_gt', default=False, action='store_true',
                        help='Whether to downsample ground truth example neighboring `def` and `return` tokens')

    args = parser.parse_args()

    device = args.device

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    writer = SummaryWriter(f'/home/shushan/modular_code_search/runs/{dt_string}')
    print("Writing to tensorboard: ", dt_string)

    if args.scorer_only:
        embedder = Embedder(device, model_eval=True)
        scorer = torch.nn.Sequential(torch.nn.Linear(embedder.dim() * 2, embedder.dim()),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(embedder.dim(), 1)).to(device)

        op = torch.optim.Adam(list(scorer.parameters()), lr=1e-8)
    else:
        embedder = Embedder(device, model_eval=False)
        scorer = torch.nn.Sequential(torch.nn.Linear(embedder.dim() * 2, embedder.dim()),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(embedder.dim(), 1)).to(device)
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

    if args.include_mismatched_pair:
        global INCLUDE_MISMATCHED_PAIR
        INCLUDE_MISMATCHED_PAIR = True

    if args.embed_separately:
        global EMBED_SEPARATELY
        EMBED_SEPARATELY = True

    if args.embed_separately:
        global DOWNSAMPLE_GT
        DOWNSAMPLE_GT = True

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
                                                       device, save_every=None,
                                                       checkpoint_prefix=checkpoint_dir + f'/model_{epoch}_ep_{i}')
        datafile_to_start = -1


if __name__ == '__main__':
    main()
