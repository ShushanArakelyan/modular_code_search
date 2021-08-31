# import numpy as np
# import pandas as pd
# import torch
#
# from scoring.embedder import Embedder
# from .utils import get_ground_truth_matches, extract_noun_tokens
# from .eval_utils import generate_HTML
#
#
# embedder = Embedder(device)
# scorer = torch.nn.Sequential(torch.nn.Linear(dim*2, dim),
#                    torch.nn.ReLU(),
#                    torch.nn.Linear(dim, 1)).to(device)
#
#
# checkpoint = '/home/shushan/finetuned_scoring_models/model_0_ep.tar'
# models = torch.load(checkpoint)
#
# scorer.load_state_dict(models['scorer'])
# embedder.model.load_state_dict(models['embedder'])
#
# input_file_name = f'/home/shushan/datasets/CodeSearchNet/resources/tagged_data_py2_py3/python/final/jsonl/train/python_train_8.jsonl.gz'
# data = pd.read_json(input_file_name, lines=True)
#
# it = 7000
#
#
# pair = 'correct_pair'
# # pair = 'incorrect_pair'
#
#
# true_scores_dfs = {}
# predicted_scores_dfs = {}
# if pair == 'correct_pair':
#     doc = data['docstring_tokens'][it]
#     code = data['alt_code_tokens'][it]
#     static_tags = data['static_tags'][it]
#     regex_tags = data['regex_tags'][it]
# noun_tokens = extract_noun_tokens(' '.join(doc))
# out_tuple = embedder.embed(doc, code, noun_tokens)
# noun_token_id_mapping, noun_token_embeddings, code_token_id_mapping, code_embedding, _, truncated_code_tokens = out_tuple
#
# # extract ground truth pairs and get scores for all pairs
# for nti, nte, nt in zip(noun_token_id_mapping, noun_token_embeddings, noun_tokens):
#     nte = nte.unsqueeze(0)
#     # check for regex and static matches
#     ground_truth_idxs = get_ground_truth_matches(nt, code, code_token_id_mapping, static_tags, regex_tags)
#
#     tiled_nte = nte.repeat(len(truncated_code_tokens), 1)
#     forward_input = torch.cat((tiled_nte, code_embedding), dim=1)
#     scorer_out = torch.sigmoid(scorer.forward(forward_input)).squeeze().cpu().detach().numpy()
#     html = generate_HTML(truncated_code_tokens, ground_truth_idxs, scorer_out)
#     # compute_f1(ground_truth_idxs, scores, split_point)
