import argparse

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from eval.dataset import transform_sample, CodeSearchNetDataset_NotPrecomputed, \
    CodeSearchNetDataset_TFIDFOracle
from eval.utils import mrr
from layout_assembly.layout import LayoutNet
from layout_assembly.modules import ScoringModule, ActionModuleFacade


def main():
    parser = argparse.ArgumentParser(description='Fine-tune codebert model for attend module')
    parser.add_argument('--scoring_checkpoint', dest='scoring_checkpoint', type=str,
                        help='load scoring model saved in checkpoint', required=True)
    parser.add_argument('--layoutnet_checkpoint', dest='layoutnet_checkpoint', type=str,
                        help='checkpoint for the action module', required=True)
    parser.add_argument('--device', dest='device', type=str,
                        help='device to run on', required=True)
    parser.add_argument('--valid_file_name', dest='valid_file_name', type=str,
                        help='validation file name', required=True)
    parser.add_argument('--action_version', dest='action_version', type=int,
                        help='Action version 1 or 2', required=True)
    parser.add_argument('--neg_sample_count', dest='neg_sample_count', type=int,
                        help='Negative samples to compare against', required=True)
    parser.add_argument('--oracle_neg_sample_count', dest='oracle_neg_sample_count', type=int,
                        help='Negative samples for oracle tf idf to output', required=True)
    parser.add_argument('--eval_version', dest='eval_version', type=str,
                        help='Version of evaluation to run, "classic" or "tf-idf"', required=True)

    args = parser.parse_args()

    device = args.device
    action_version = args.action_version
    scoring_module = ScoringModule(device, checkpoint=args.scoring_checkpoint, eval=True)
    action_module = ActionModuleFacade(device, action_version, eval=True)
    layout_net = LayoutNet(scoring_module, action_module, device, precomputed_scores_provided=False, eval=True)
    layout_net.load_from_checkpoint(args.layoutnet_checkpoint)

    neg_sample_count = args.neg_sample_count
    oracle_neg_sample_count = args.oracle_neg_sample_count
    valid_file_name = args.valid_file_name
    eval_version = args.eval_version
    print("Evaluation ", eval_version)
    if eval_version == 'classic':
        dataset = CodeSearchNetDataset_NotPrecomputed(valid_file_name, device, neg_sample_count)
    elif eval_version == 'tf-idf':
        dataset = CodeSearchNetDataset_TFIDFOracle(valid_file_name, device, neg_sample_count, oracle_neg_sample_count)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    MRRs = []
    with torch.no_grad():
        i = 0
        for samples in tqdm.tqdm(data_loader):
            i += 1
            ranks = []
            sample = samples[0]
            pred = layout_net.forward(*transform_sample(sample))
            if pred:
                ranks.append(torch.sigmoid(pred).cpu().numpy())
            else:
                continue
            for sample in samples[1:]:
                pred = layout_net.forward(*transform_sample(sample))
                if pred:
                    ranks.append(torch.sigmoid(pred).cpu().numpy())
                else:
                    ranks.append(np.random.rand(1)[0])
            MRRs.append(mrr(ranks))
            if i % 100 == 0:
                print(np.mean(MRRs))
        print(np.mean(MRRs))


if __name__ == "__main__":
    main()
