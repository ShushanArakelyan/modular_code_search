import argparse
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from scoring.embedder import Embedder
from scoring.eval_utils import find_split_point, eval_example


def run_eval_epoch(data, scorer, embedder, split_point=0.5):
    f1_scores = []
    precisions = []
    recalls = []
    for it in tqdm(range(len(data))):
        result_dict = eval_example(data, it, scorer, embedder, evaluate="F1", split_point=split_point)
        f1_scores.append(np.mean(result_dict['f1_scores_for_sample']))
        precisions.append(np.mean(result_dict['pre_for_sample']))
        recalls.append(np.mean(result_dict['re_for_sample']))
    return f1_scores, precisions, recalls


def main():
    parser = argparse.ArgumentParser(description='Fine-tune codebert model for attend module')
    parser.add_argument('--load_from_checkpoint', dest='checkpoint', type=str,
                        help='continue training from checkpoint', required=True)
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on', required=True)
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='validation data directory', required=True)
    parser.add_argument('--train_file_name', dest='train_file_name', type=str,
                        help='data to find the best split point', required=True)
    args = parser.parse_args()

    device = args.device
    embedder = Embedder(device, model_eval=True)
    scorer = torch.nn.Sequential(torch.nn.Linear(embedder.get_dim() * 2, embedder.get_dim()),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(embedder.get_dim(), 1)).to(device)
    print(f"Evaluating checkpoint {args.checkpoint}")
    models = torch.load(args.checkpoint, map_location=device)
    scorer.load_state_dict(models['scorer'])
    scorer = scorer.to(device)
    embedder.model.load_state_dict(models['embedder'])
    embedder.model = embedder.model.to(device)

    train_data = pd.read_json(args.train_file_name, lines=True)
    print("Searching for best split point on train data....")
    with torch.no_grad():
        split_points, f1_scores = find_split_point(train_data[:50], scorer, embedder)
        split_point = split_points[np.argmax(f1_scores)]
        print(f"Search complete, will be using split_point={split_point}")
        valid_data = pd.read_json(args.valid_file_name, lines=True)
        print("Running evaluation....")
        f1_scores, precisions, recalls = run_eval_epoch(valid_data, scorer, embedder, split_point=split_point)
        print(f"Mean precision: {np.mean(precisions)}, mean recall: {np.mean(recalls)}, mean F1: {np.mean(f1_scores)}")


if __name__ == '__main__':
    main()
