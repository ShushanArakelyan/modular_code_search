# the code in this file was designed to generate pairwise data for codebert finetuning;
# however, since we might want to finetune differently than simply using a pair of tokens
# the code has been rewritten and reformatted in other files.


# import sys
#
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# from scoring.utils import P, extract_noun_tokens, filter_embedding_by_id, get_word_to_roberta_tokens, \
#     get_orig_tokens_to_roberta_tokens, get_matched_labels_binary_v2, get_static_labels_binary, get_regex_labels_binary
# from .embedder import Embedder
#
#
# def generate_finetuning_data(embedder, data, output_file):
#     res_data = []
#     res_scores = []
#     part = 0
#
#     for it in tqdm(range(len(data)), total=len(data), desc="Row: "):
#         # sample some query and some code, half the cases will have the correct pair, the other half the cases will have an incorrect pair
#         for pair in ['correct_pair', 'incorrect_pair']:
#             if pair == 'correct_pair':
#                 doc = data['docstring_tokens'][it]
#                 code = data['alt_code_tokens'][it]
#                 static_tags = data['static_tags'][it]
#                 regex_tags = data['regex_tags'][it]
#             else:
#                 # to make this reproducible
#                 np.random.seed(it)
#                 random_idx = np.random.choice(np.arange(len(data)), 1)[0]
#                 doc = data['docstring_tokens'][it]
#                 code = data['alt_code_tokens'][random_idx]
#                 static_tags = data['static_tags'][random_idx]
#                 regex_tags = data['regex_tags'][random_idx]
#             if len(doc) == 0 or len(code) == 0:
#                 continue
#             noun_tokens = extract_noun_tokens(' '.join(doc))
#             # converting the docstring and code tokens into CodeBERT inputs
#             # CodeBERT inputs are limited by 512 tokens, so this will truncate the inputs
#             inputs = embedder.get_feature_inputs(' '.join(doc), ' '.join(code))
#             separator = np.where(inputs['input_ids'][0].cpu().numpy() == embedder.tokenizer.sep_token_id)[0][0]
#             # ignore CLS tokens at the beginning and at the end
#             query_token_ids, code_token_ids = inputs['input_ids'][0][1:separator], inputs['input_ids'][0][
#                                                                                    separator + 1:-1]
#
#             # get truncated version of code and query
#             truncated_code_tokens = embedder.tokenizer.convert_ids_to_tokens(code_token_ids)
#             truncated_query_tokens = embedder.tokenizer.convert_ids_to_tokens(query_token_ids)
#
#             # mapping from CodeBERT tokenization to our dataset tokenization
#             noun_token_id_mapping = np.asarray(get_word_to_roberta_tokens(doc,
#                                                                           truncated_query_tokens,
#                                                                           noun_tokens, embedder), dtype=object)
#
#             code_token_id_mapping = np.asarray(get_orig_tokens_to_roberta_tokens(code,
#                                                                           truncated_code_tokens,
#                                                                           embedder), dtype=object)
#             if noun_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
#                 continue
#
#             # get CodeBERT embedding for the example
#             embedding = embedder.get_embeddings(inputs).cpu().detach().numpy()
#             query_embedding, code_embedding = embedding[1:separator], embedding[separator + 1:-1]
#             noun_token_embeddings = filter_embedding_by_id(query_embedding, noun_token_id_mapping)
#
#             # extract positive pairs and sample negative pairs
#             for nti, nte, nt in zip(noun_token_id_mapping, noun_token_embeddings, noun_tokens):
#                 # check for regex and static matches
#                 pos_sample_idxs = []
#                 idxs = \
#                 get_matched_labels_binary_v2(code[:len(code_token_id_mapping)], code_token_id_mapping, nt).nonzero()[0]
#                 if idxs.size:
#                     pos_sample_idxs.extend(idxs)
#                 idxs = get_static_labels_binary(code_token_id_mapping, nt, static_tags).nonzero()[0]
#                 if idxs.size:
#                     pos_sample_idxs.extend(idxs)
#                 idxs = get_regex_labels_binary(code_token_id_mapping, nt, regex_tags).nonzero()[0]
#                 if idxs.size:
#                     pos_sample_idxs.extend(idxs)
#                 # add positive example
#                 if len(pos_sample_idxs) > 0:
#                     pos_sample_idxs = np.unique(pos_sample_idxs)
#                     pos_samples_code_emb = code_embedding[pos_sample_idxs]
#                     tiled_nte = np.tile(nte, (pos_samples_code_emb.shape[0], 1))
#                     res_data.extend(np.hstack((tiled_nte, pos_samples_code_emb)))
#                     res_scores.extend(np.ones(pos_samples_code_emb.shape[0]))
#
#                 # sample random number of negative examples
#                 num_neg_samples = np.sum(np.random.binomial(n=10, p=P))
#                 unique_ids, counts = np.unique(code[:len(code_token_id_mapping)], return_counts=True)
#                 id_freq_dict = {uid: c for uid, c in zip(unique_ids, counts)}
#                 p = np.asarray([1 / id_freq_dict[i] for i in code[:len(code_token_id_mapping)]])
#                 p = p / np.sum(p)
#                 neg_sample_idxs = np.random.choice(np.arange(len(code_token_id_mapping)), num_neg_samples,
#                                                    replace=False, p=p)
#                 neg_sample_idxs = np.delete(neg_sample_idxs, np.where(neg_sample_idxs in pos_sample_idxs)[0])
#
#                 attempt = 0
#                 while neg_sample_idxs.size == 0 and attempt < 5:
#                     attempt += 1
#                     neg_sample_idxs = np.random.choice(np.arange(len(code_token_id_mapping)), num_neg_samples,
#                                                        replace=False, p=p)
#                     neg_sample_idxs = np.delete(neg_sample_idxs, np.where(neg_sample_idxs in pos_sample_idxs)[0])
#                 if attempt == 5:
#                     continue
#
#                 neg_sample_code_emb = filter_embedding_by_id(code_embedding, code_token_id_mapping[neg_sample_idxs])
#                 tiled_nte = np.tile(nte, (neg_sample_code_emb.shape[0], 1))
#                 res_data.extend(np.hstack((tiled_nte, neg_sample_code_emb)))
#                 res_scores.extend(np.zeros(len(neg_sample_code_emb)))
#                 if not np.all(np.asarray([len(r) for r in res_data[-1000:]]) == 1536):
#                     raise Exception
#                 if len(res_data) >= 1000000:
#                     res_data = np.asarray(res_data)
#                     res_scores = np.asarray(res_scores)
#                     np.save(f'{output_file}_data_{part}.npy', res_data)
#                     np.save(f'{output_file}_scores_{part}.npy', res_scores)
#                     part += 1
#                     res_data = []
#                     res_scores = []
#     res_data = np.asarray(res_data)
#     res_scores = np.asarray(res_scores)
#     if part > 0:
#         np.save(f'{output_file}_data_{part}.npy', res_data)
#         np.save(f'{output_file}_scores_{part}.npy', res_scores)
#     else:
#         np.save(f'{output_file}_data.npy', res_data)
#         np.save(f'{output_file}_scores.npy', res_scores)
#     return res_data, res_scores
#
#
# def main():
#     input_file = sys.argv[1]
#     output_file = sys.argv[2]
#     if len(sys.argv) > 3:
#         device = sys.argv[3]
#     else:
#         device = None
#     data = pd.read_json(input_file, lines=True)
#     embedder = Embedder(device)
#     train_data, train_labels = generate_finetuning_data(embedder, data, output_file)
#
#
# if __name__ == '__main__':
#     main()
