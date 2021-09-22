import torch

from itertools import chain

import codebert_embedder_with_adapter as embedder
from layout_assembly.action_v1 import ActionModule_v1
from layout_assembly.utils import ProcessingException


class ActionModule_v1_reduced(ActionModule_v1):
    def __init__(self, device):
        ActionModule_v1.__init__(self, device)
        self.reduce_linear = None

    def parameters(self):
        return chain(self.model1.parameters(), self.model2.parameters())

#     def load_state_dict(self, d):
#         self.model1.load_state_dict(d['model1'])
#         self.model1 = self.model1.to(self.device)
#         self.model2.load_state_dict(d['model2'])
#         self.model2 = self.model2.to(self.device)

#     def state_dict(self):
#         return {'model1': self.model1.state_dict(), 'model2': self.model2.state_dict()}

#     def eval(self):
#         self.model1.eval()
#         self.model2.eval()

#     def train(self):
#         self.model1.train()
#         self.model2.train()



class ActionModule_v1_reduced_one_input(ActionModule_v1_reduced):
    def __init__(self, device, eval=False):
        ActionModule_v1_reduced.__init__(self, device)
        dim = embedder.dim
#         reduced_dim = 32
#         self.reduce_linear = torch.nn.Linear(embedder.dim, reduced_dim).to(self.device)
        # outputs a sequence of scores
        self.model1 = torch.nn.Sequential(torch.nn.Linear(dim * 2 + 1, 128),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, 1)).to(self.device)
        # outputs an embedding
        self.model2 = torch.nn.Sequential(torch.nn.Linear(dim + embedder.max_seq_length, 128),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, dim)).to(self.device)

    def forward(self, _, arg1, __, precomputed_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            prep_embedding = (scores[0] + prep_embedding) / 2
            scores = scores[1]
#         prep_embedding = self.reduce_linear(prep_embedding)
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)
            
        if precomputed_embeddings is None:
            raise ProcessingException()
        _, code_embeddings = precomputed_embeddings
        
        scores_out = self.model1.forward(torch.cat(
            (prep_embedding.repeat(embedder.max_seq_length, 1),
             code_embeddings,
             scores), dim=1))
        emb_out = self.model2.forward(
            torch.cat((prep_embedding, scores_out.squeeze().unsqueeze(dim=0)), dim=1))
        return emb_out, scores_out


class ActionModule_v1_reduced_two_inputs(ActionModule_v1_reduced):
    def __init__(self, device, eval=False):
        ActionModule_v1_reduced.__init__(self, device)
        dim = embedder.dim        
#         reduced_dim = 32
#         self.reduce_linear = torch.nn.Linear(embedder.dim, reduced_dim).to(self.device)
        # outputs a sequence of scores
        self.model1 = torch.nn.Sequential(torch.nn.Linear(dim * 2 + dim + 2, 128),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, 1)).to(self.device)
        # outputs an embedding
        self.model2 = torch.nn.Sequential(torch.nn.Linear(dim * 2 + embedder.max_seq_length, 128),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, dim)).to(self.device)

    def forward(self, _, args, __, precomputed_embeddings):
        arg1, arg2 = args
        prep1_embedding, scores1 = arg1
        if isinstance(scores1, tuple):
            prep1_embedding = (scores1[0] + prep1_embedding) / 2
            scores1 = scores1[1]
        if len(scores1.shape) == 1:
            scores1 = scores1.unsqueeze(dim=1)
        prep2_embedding, scores2 = arg2
        if isinstance(scores2, tuple):
            prep2_embedding = (scores2[0] + prep2_embedding) / 2
            scores2 = scores2[1]
        if len(scores2.shape) == 1:
            scores2 = scores2.unsqueeze(dim=1)
#         prep1_embedding = self.reduce_linear(prep1_embedding)
#         prep2_embedding = self.reduce_linear(prep2_embedding)
        
        if precomputed_embeddings is None:
            raise ProcessingException()
        _, code_embeddings = precomputed_embeddings
        scores_out = self.model1.forward(torch.cat(
            (prep1_embedding.repeat(embedder.max_seq_length, 1),
             prep2_embedding.repeat(embedder.max_seq_length, 1),
             code_embeddings,
             scores1,
             scores2), dim=1))
        emb_out = self.model2.forward(torch.cat((prep1_embedding,
                                                 prep2_embedding,
                                                 scores_out.squeeze().unsqueeze(dim=0)), dim=1))
        return emb_out, scores_out