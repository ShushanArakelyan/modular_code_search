# import numpy as np
# import spacy
# import torch
#
# from third_party.CodeBERT.CodeBERT.codesearch.utils import InputExample
# from .embedder import Embedder
# from .model import ScoringLayer
# from .process import pos_tagging
#
# nlp = spacy.load("en_core_web_md")
#
#
# class Scorer(object):
#     """This class is used to assess the embeddings of the query and code tokens
#     after training the classification layer."""
#
#     def __init__(self, device, model_checkpoint=None, train_checkpoint=None, dim=768):
#         self.embedder = Embedder(device, model_checkpoint)
#         self.scoring_layer = ScoringLayer(dim)
#         self.loss = torch.nn.BCELoss()
#         self.scoring_layer.load_state_dict(torch.load(train_checkpoint)['model_state_dict'])
#
#     def forward(self, data):
#         query, code = data
#         cexample = [InputExample(0, code, label="0")]
#         qexample = [InputExample(0, query, label="0")]
#
#         cinputs = self.embedder.get_feature_inputs(cexample)
#         cembeddings = self.embedder.get_embeddings(cinputs)
#         cembeddings = cembeddings.detach().cpu().numpy()
#
#         qinputs = self.embedder.get_feature_inputs(qexample)
#         qembeddings = self.embedder.get_embeddings(qinputs)
#         qtokens = pos_tagging(query)
#
#         score_matrix = np.zeros((len(qtokens), cembeddings.shape[0], cembeddings.shape[1] + qembeddings.shape[1]))
#         for i, qtoken in enumerate(qtokens):
#             qtoken_embedding = self.embedder.get_token_embedding(qinputs, qembeddings, qtoken)
#             qtoken_embedding = qtoken_embedding.detach().cpu().numpy()
#
#             # only tile non-zero code embeddings
#             qtoken_embedding = np.tile(qtoken_embedding, (len(cembeddings), 1))
#             pair_embedding = np.concatenate((qtoken_embedding, cembeddings), axis=1)
#
#             score_matrix[i] = pair_embedding
#         return score_matrix
#
#     def forward(self, data):
#         data = self.preprocess(data)
#         return self.scoring_layer.forward(data)
