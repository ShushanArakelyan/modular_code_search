import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import scoring as embedder
from scoring.scoring_eval_utils import find_split_point, eval_example


def run_eval_epoch(data, scorer, embedder, embed_separately, version, normalize, split_point=0.5):
    f1_scores = []
    precisions = []
    recalls = []
    for it in tqdm(range(len(data))):
        result_dict = eval_example(data, it, scorer, embedder, evaluate="F1",
                                   split_point=split_point, embed_separately=embed_separately, version=version,
                                   normalize=normalize)
        f1 = result_dict['f1_scores_for_sample']
        pre = result_dict['pre_for_sample']
        re = result_dict['re_for_sample']
        if len(f1) > 0 and len(pre) > 0 and len(re) > 0:
            f1_scores.append(np.mean(f1))
            precisions.append(np.mean(pre))
            recalls.append(np.mean(re))
    return f1_scores, precisions, recalls


def main():
    parser = argparse.ArgumentParser(description='Fine-tune codebert model for attend module')
    parser.add_argument('--load_from_checkpoint', dest='checkpoint', type=str,
                        help='evaluate model saved in checkpoint')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str,
                        help='evaluate model on all checkpoints in the directory')
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on', required=True)
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='validation file name', required=True)
    parser.add_argument('--embed_separately', dest='embed_separately', default=False, action='store_true',
                        help='whether to embed the query and code separately or concatenated together')
    parser.add_argument('--normalize', dest='normalize', default=False, action='store_true',
                        help='whether to normalize the predictions of the model using random tokens')
    parser.add_argument('--version', dest='version', type=str,
                        help='how to compute query embedding, can be CLS or MEAN', default="CLS")

    args = parser.parse_args()
    checkpoints = []
    assert (args.checkpoint or args.checkpoint_dir), \
        "Either --load_from_checkpoint or --checkpoint_dir must be specified"
    assert (not (args.checkpoint and args.checkpoint_dir)), \
        "Only one from --load_from_checkpoint or --checkpoint_dir can be specified"

    if args.checkpoint is not None:
        checkpoints.append(args.checkpoint)
    elif args.checkpoint_dir:
        import glob
        all_checkpoints = glob.glob(args.checkpoint_dir + '/*')
        for checkpoint in all_checkpoints:
            assert checkpoint.endswith('.tar')
            checkpoints.append(checkpoint)

    device = args.device
    version = args.version
    normalize = args.normalize
    embed_separately = args.embed_separately
    valid_data = pd.read_json(args.valid_file_name, lines=True)

    with torch.no_grad():
        for checkpoint in checkpoints:
            if not embedder.initialized:
                embedder.init_embedder(device)
            scorer = torch.nn.Sequential(torch.nn.Linear(embedder.dim * 2, embedder.dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(embedder.dim, 1)).to(device)

            print(f"Evaluating checkpoint {checkpoint}")
            models = torch.load(checkpoint, map_location=device)
            scorer.load_state_dict(models['scorer'])
            scorer = scorer.to(device)
            embedder.model.load_state_dict(models['embedder'])
            embedder.model = embedder.model.to(device)

            print("Searching for best split point on valid data....")
            split_points, f1_scores = find_split_point(valid_data[:50], scorer, embedder, embed_separately, version,
                                                       normalize)
            split_point = split_points[np.argmax(f1_scores)]
            print(f"Search complete, will be using split_point={split_point}")
            print("Running evaluation....")
            f1_scores, precisions, recalls = run_eval_epoch(valid_data, scorer, embedder, embed_separately, version,
                                                            normalize=normalize,
                                                            split_point=split_point)
            print(f"Mean precision: {np.mean(precisions)}, ", f"mean recall: {np.mean(recalls)}, ",
                  f"mean F1: {np.mean(f1_scores)}")


if __name__ == '__main__':
    main()
