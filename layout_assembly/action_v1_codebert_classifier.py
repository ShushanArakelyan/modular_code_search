from itertools import chain

import torch

import codebert_embedder_v2 as embedder
from layout_assembly.utils import ProcessingException, FC2, FC2_normalized, init_weights


class ActionModule_v1:
    def __init__(self, device, normalize=False):
        self.device = device
        if not embedder.initialized:
            embedder.init_embedder(device)
        self.model1 = None
        self.normalized = normalize
        self.init_networks()

    def init_networks(self):
        raise Exception("Not Implemented")

    def parameters(self):
        return self.model1.parameters()

    def named_parameters(self):
        return self.model1.named_parameters()

    def load_state_dict(self, d):
        self.model1.load_state_dict(d['model1'])
        self.model1 = self.model1.to(self.device)

    def state_dict(self):
        return {'model1': self.model1.state_dict()}

    def eval(self):
        self.model1.eval()
        embedder.classifier.eval()

    def train(self):
        self.model1.train()
        embedder.classifier.train()


class ActionModule_v1_one_input(ActionModule_v1):
    def init_networks(self):
        hidden_input_dims = [embedder.dim * 3 + 1, 512]
        hidden_output_dims = [512, 1]
        # outputs a sequence of scores
        if self.normalized:
            self.model1 = FC2_normalized(hidden_input_dims, hidden_output_dims).to(self.device)
        else:
            self.model1 = FC2(hidden_input_dims, hidden_output_dims).to(self.device)
        self.model1.apply(init_weights)

    def forward(self, _, arg1, __, precomputed_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            _, scores = scores
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)

        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings, inputs = precomputed_embeddings
        tiled_verb_emb = verb_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep_emb = prep_embedding.repeat(embedder.max_seq_length, 1)
        model1_input = torch.cat((tiled_verb_emb, tiled_prep_emb, code_embeddings, scores), dim=1)
        scores_out = self.model1.forward(model1_input)
        l1_reg_loss = torch.norm(scores_out, 1)
        return None, scores_out, l1_reg_loss


class ActionModule_v1_two_inputs(ActionModule_v1):
    def init_networks(self):
        hidden_input_dims = [embedder.dim * 4 + 2, 512]
        hidden_output_dims = [512, 1]
        if self.normalized:
            self.model1 = FC2_normalized(hidden_input_dims, hidden_output_dims).to(self.device)
        else:
            self.model1 = FC2(hidden_input_dims, hidden_output_dims).to(self.device)
        self.model1.apply(init_weights)

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
        verb_embedding, code_embeddings, inputs = precomputed_embeddings
        tiled_verb_emb = verb_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep1_emb = prep1_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep2_emb = prep2_embedding.repeat(embedder.max_seq_length, 1)
        model1_input = torch.cat(
            (tiled_verb_emb, tiled_prep1_emb, tiled_prep2_emb, code_embeddings, scores1, scores2), dim=1)
        scores_out = self.model1.forward(model1_input)
        l1_reg_loss = torch.norm(scores_out, 1)
        return None, scores_out, l1_reg_loss
