from itertools import chain

import torch
import torch.nn as nn

import codebert_embedder_v2 as embedder
from layout_assembly.utils import ProcessingException, FC2, FC2_normalized, init_weights


class ActionModule_v5:
    def __init__(self, device, normalize=False, eval=False):
        self.device = device
        if not embedder.initialized:
            embedder.init_embedder(device)
        self.encoder_layer = None
        self.mlp = None
        self.normalized = normalize
        self.is_eval = eval
        self.init_networks()

    def init_networks(self):
        raise Exception("Not Implemented")

    def parameters(self):
        return chain(self.encoder_layer.parameters(), self.mlp.parameters())

    def named_parameters(self):
        return chain(self.encoder_layer.named_parameters(), self.mlp.named_parameters())

    def load_state_dict(self, d):
        self.encoder_layer.load_state_dict(d['encoder_layer'])
        self.encoder_layer = self.encoder_layer.to(self.device)
        self.mlp.load_state_dict(d['mlp'])
        self.mlp = self.mlp.to(self.device)

    def state_dict(self):
        return {'encoder_layer': self.encoder_layer.state_dict(), 'mlp': self.mlp.state_dict()}

    def eval(self):
        self.encoder_layer.eval()
        self.mlp.eval()

    def train(self):
        self.encoder_layer.train()
        self.mlp.train()


class ActionModule_v5_one_input(ActionModule_v5):
    def init_networks(self):
        # outputs a sequence of scores
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedder.dim, nhead=8).to(self.device)
        hidden_input_dims = [embedder.dim, 512]
        hidden_output_dims = [512, 1]
        if self.normalized:
            self.mlp = FC2_normalized(hidden_input_dims, hidden_output_dims).to(self.device)
        else:
            self.mlp = FC2(hidden_input_dims, hidden_output_dims).to(self.device)
        self.mlp.apply(init_weights)
        if self.is_eval:
            self.eval()

    def forward(self, _, arg1, __, precomputed_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            _, scores = scores
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)
        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        # here the code embeddings have 1 more sep symbol at the beginning
        weighted_code_emb = torch.mm(scores.T, code_embeddings[1:]) 
        encoder_input = torch.cat((embedder.cls_embedding, verb_embedding, 
                                   prep_embedding, weighted_code_emb, code_embeddings)).unsqueeze(dim=1)
#         print("encoder input: ", encoder_input.shape)
        mlp_input = self.encoder_layer(encoder_input)
#         print("mlp input: ", mlp_input.shape)
        mlp_input = torch.index_select(mlp_input, 0, torch.LongTensor(range(5, len(encoder_input))).to(self.device)).squeeze()
#         print("mlp input: ", mlp_input.shape)
        scores_out = self.mlp.forward(mlp_input)
        l1_reg_loss = torch.norm(scores_out, 1)
        return None, scores_out, l1_reg_loss


class ActionModule_v5_two_inputs(ActionModule_v5):
    def init_networks(self):
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedder.dim, nhead=8).to(self.device)
        hidden_input_dims = [embedder.dim, 512]
        hidden_output_dims = [512, 1]
        if self.normalized:
            self.mlp = FC2_normalized(hidden_input_dims, hidden_output_dims).to(self.device)
        else:
            self.mlp = FC2(hidden_input_dims, hidden_output_dims).to(self.device)
        self.mlp.apply(init_weights)
        if self.is_eval:
            self.eval()

    def forward(self, _, args, __, precomputed_embeddings):
        arg1, arg2 = args
        prep1_embedding, scores1 = arg1
        if isinstance(scores1, tuple):
            _, scores1 = scores1
        if len(scores1.shape) == 1:
            scores1 = scores1.unsqueeze(dim=1)
        prep2_embedding, scores2 = arg2
        if isinstance(scores2, tuple):
            _, scores2 = scores2
        if len(scores2.shape) == 1:
            scores2 = scores2.unsqueeze(dim=1)

        if precomputed_embeddings is None:
            raise ProcessingException()
            
        verb_embedding, code_embeddings = precomputed_embeddings
#         print(code_embeddings[1:].shape, scores1.shape)
        weighted_code_emb1 = torch.mm(scores1.T, code_embeddings[1:])
        weighted_code_emb2 = torch.mm(scores2.T, code_embeddings[1:])
        encoder_input = torch.cat((embedder.cls_embedding, verb_embedding, prep1_embedding, 
                                   prep2_embedding, weighted_code_emb1, 
                                   weighted_code_emb2, code_embeddings)).unsqueeze(dim=1)
#         print("encoder input: ", encoder_input.shape)
        mlp_input = self.encoder_layer(encoder_input)
#         print("mlp input: ", mlp_input.shape)
        mlp_input = torch.index_select(mlp_input, 0, torch.LongTensor(range(7, len(encoder_input))).to(self.device)).squeeze()
#         print("mlp input: ", mlp_input.shape)
        scores_out = self.mlp.forward(mlp_input)
        l1_reg_loss = torch.norm(scores_out, 1)
        return None, scores_out, l1_reg_loss
