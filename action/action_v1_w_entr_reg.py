from itertools import chain

import torch

import codebert_embedder as embedder
from layout_assembly.utils import ProcessingException, FC2, FC2_normalized, init_weights


class ActionModule_v1:
    def __init__(self, device, normalize=False, dropout=0):
        self.device = device
        if not embedder.initialized:
            embedder.init_embedder(device)
        self.model1 = None
        self.model2 = None
        self.dropout = dropout
        self.normalized = normalize
        self.init_networks()

    def init_networks(self):
        raise Exception("Not Implemented")

    def parameters(self):
        return chain(self.model1.parameters(), self.model2.parameters())

    def named_parameters(self):
        return chain(self.model1.named_parameters(), self.model2.named_parameters())

    def load_state_dict(self, d):
        self.model1.load_state_dict(d['model1'])
        self.model1 = self.model1.to(self.device)
        self.model2.load_state_dict(d['model2'])
        self.model2 = self.model2.to(self.device)

    def state_dict(self):
        return {'model1': self.model1.state_dict(), 'model2': self.model2.state_dict()}

    def eval(self):
        self.model1.eval()
        self.model2.eval()
        embedder.model.eval()

    def train(self):
        self.model1.train()
        self.model2.train()
        embedder.model.train()


class ActionModule_v1_one_input(ActionModule_v1):
    def init_networks(self):
        hidden_input_dims = [embedder.dim * 3 + 1, 512]
        hidden_output_dims = [512, 1]
        # outputs a sequence of scores
        if self.normalized:
            self.model1 = FC2_normalized(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        else:
            self.model1 = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        self.model1.apply(init_weights)
        #         hidden_input_dims = [embedder.dim * 2 + embedder.max_seq_length, 128]
        #         hidden_output_dims = [128, embedder.dim]
        hidden_input_dims = [embedder.max_seq_length, 512]
        hidden_output_dims = [512, embedder.dim]
        # outputs an embedding
        self.model2 = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        self.model2.apply(init_weights)

    def forward(self, _, arg1, __, precomputed_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            emb, scores = scores
            prep_embedding = (emb + prep_embedding) / 2
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)

        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        tiled_verb_emb = verb_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep_emb = prep_embedding.repeat(embedder.max_seq_length, 1)
        model1_input = torch.cat((tiled_verb_emb, tiled_prep_emb, code_embeddings, scores), dim=1)
        scores_out = self.model1.forward(model1_input)
        #         model2_input = torch.cat((verb_embedding, prep_embedding, scores_out.squeeze().unsqueeze(dim=0)), dim=1)
        model2_input = scores_out.squeeze().unsqueeze(dim=0)
        emb_out = self.model2.forward(model2_input)
        entr_reg_loss = -torch.matmul(scores_out.T, torch.log(scores_out)).mean()
        return emb_out, scores_out, entr_reg_loss


class ActionModule_v1_two_inputs(ActionModule_v1):
    def init_networks(self):
        hidden_input_dims = [embedder.dim * 4 + 2, 512]
        hidden_output_dims = [512, 1]
        if self.normalized:
            self.model1 = FC2_normalized(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        else:
            self.model1 = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        self.model1.apply(init_weights)
        #         hidden_input_dims = [embedder.dim * 3 + embedder.max_seq_length, 128]
        #         hidden_output_dims = [128, embedder.dim]
        hidden_input_dims = [embedder.max_seq_length, 512]
        hidden_output_dims = [512, embedder.dim]
        self.model2 = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        self.model2.apply(init_weights)

    def forward(self, _, args, __, precomputed_embeddings):
        arg1, arg2 = args
        prep1_embedding, scores1 = arg1
        if isinstance(scores1, tuple):
            emb, scores1 = scores1
            prep1_embedding = (emb + prep1_embedding) / 2
        if len(scores1.shape) == 1:
            scores1 = scores1.unsqueeze(dim=1)
        prep2_embedding, scores2 = arg2
        if isinstance(scores2, tuple):
            emb, scores2 = scores2
            prep2_embedding = (emb + prep2_embedding) / 2
        if len(scores2.shape) == 1:
            scores2 = scores2.unsqueeze(dim=1)

        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        tiled_verb_emb = verb_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep1_emb = prep1_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep2_emb = prep2_embedding.repeat(embedder.max_seq_length, 1)
        model1_input = torch.cat(
            (tiled_verb_emb, tiled_prep1_emb, tiled_prep2_emb, code_embeddings, scores1, scores2), dim=1)
        scores_out = self.model1.forward(model1_input)
        #         model2_input = torch.cat(
        #             (verb_embedding, prep1_embedding, prep2_embedding, scores_out.squeeze().unsqueeze(dim=0)),
        #             dim=1)
        model2_input = scores_out.squeeze().unsqueeze(dim=0)
        emb_out = self.model2.forward(model2_input)
        entr_reg_loss = -torch.matmul(scores_out.T, torch.log(scores_out)).mean()
        return emb_out, scores_out, entr_reg_loss
