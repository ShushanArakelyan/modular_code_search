from itertools import chain

import torch

import codebert_embedder as embedder
from layout_assembly.action_v1 import ActionModule_v1
from layout_assembly.utils import ProcessingException, FC2, FC2_normalized


class ActionModule_v3(ActionModule_v1):
    def __init__(self, device, normalize=False, eval=False, dropout=0):
        ActionModule_v1.__init__(self, device, normalize, eval)
        self.attention = None
        self.dropout = dropout
        self.init_networks()

    def parameters(self):
        return chain(self.model1.parameters(), self.model2.parameters(), self.attention.parameters())

    def named_parameters(self):
        return chain(self.model1.named_parameters(), self.model2.named_parameters(), self.attention.named_parameters())

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.attention.load_state_dict(d['attention'])
        self.attention = self.attention.to(self.device)

    def state_dict(self):
        d = super().state_dict()
        d['attention'] = self.attention.state_dict()
        return d

    def eval(self):
        super().eval()
        self.attention.eval()

    def train(self):
        super().train()
        self.attention.train()


class ActionModule_v3_one_input(ActionModule_v3):
    def init_networks(self):
        # attentions for the sequence
        hidden_input_dims = [embedder.dim + 2, 64]
        hidden_output_dims = [64, embedder.max_seq_length]
        self.attention = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        # outputs a sequence of scores
        hidden_input_dims = [embedder.dim * 2, 128]
        hidden_output_dims = [128, 1]
        if self.normalized:
            self.model1 = FC2_normalized(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        else:
            self.model1 = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)

        #         hidden_input_dims = [embedder.dim * 2 + embedder.max_seq_length, 128]
        #         hidden_output_dims = [128, embedder.dim]
        hidden_input_dims = [embedder.max_seq_length, 128]
        hidden_output_dims = [128, embedder.dim]
        # outputs an embedding
        self.model2 = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)

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
        weights = self.attention(
            torch.cat((tiled_prep_emb, scores, torch.arange(embedder.max_seq_length).unsqueeze(dim=1).to(self.device)), dim=1))
#         print(weights.shape)  # 512x512
        weights = weights.unsqueeze(dim=1)
#         print(weights.shape)  # 512x1x512
        weighted_code_embeddings = torch.matmul(weights, code_embeddings)
#         print(weighted_code_embeddings.shape)  # 512x1x768
        weighted_code_embeddings = weighted_code_embeddings.squeeze()  # 512x768
        model1_input = torch.cat((tiled_verb_emb, weighted_code_embeddings), dim=1)
        scores_out = self.model1.forward(model1_input)
        #         model2_input = torch.cat((verb_embedding, prep_embedding, scores_out.squeeze().unsqueeze(dim=0)), dim=1)
        model2_input = scores_out.squeeze().unsqueeze(dim=0)
        emb_out = self.model2.forward(model2_input)
        l1_reg_loss = torch.norm(scores_out, 1)
        return emb_out, scores_out, l1_reg_loss


class ActionModule_v3_two_inputs(ActionModule_v3):
    def init_networks(self):
        # attentions for the sequence
        hidden_input_dims = [embedder.dim + 2, 64]
        hidden_output_dims = [64, embedder.max_seq_length]
        self.attention = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        # outputs a sequence of scores
        hidden_input_dims = [embedder.dim * 3, 128]
        hidden_output_dims = [128, 1]
        if self.normalized:
            self.model1 = FC2_normalized(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        else:
            self.model1 = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        #         hidden_input_dims = [embedder.dim * 3 + embedder.max_seq_length, 128]
        #         hidden_output_dims = [128, embedder.dim]
        hidden_input_dims = [embedder.max_seq_length, 128]
        hidden_output_dims = [128, embedder.dim]
        self.model2 = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)

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
#         print(tiled_prep1_emb.shape, scores1.shape, torch.arange(embedder.max_seq_length).unsqueeze(dim=1).shape)
        weights_prep1 = self.attention.forward(
            torch.cat(
                (tiled_prep1_emb, scores1, torch.arange(embedder.max_seq_length).unsqueeze(dim=1).to(self.device)), dim=1))
        weights_prep2 = self.attention.forward(
            torch.cat((tiled_prep2_emb, scores2, torch.arange(embedder.max_seq_length).unsqueeze(dim=1).to(self.device)), dim=1))
#         print(weights_prep1.shape)  # 512x512
        weights_prep1 = weights_prep1.unsqueeze(dim=1)
        weights_prep2 = weights_prep2.unsqueeze(dim=1)
#         print(weights_prep2.shape)  # 512x1x512
        weighted_code_embeddings_prep1 = torch.matmul(weights_prep1, code_embeddings)
        weighted_code_embeddings_prep2 = torch.matmul(weights_prep2, code_embeddings)
#         print(weighted_code_embeddings_prep2.shape)  # 512x1x768
        weighted_code_embeddings_prep1 = weighted_code_embeddings_prep1.squeeze()  # 512x768
        weighted_code_embeddings_prep2 = weighted_code_embeddings_prep2.squeeze()  # 512x768
        model1_input = torch.cat((tiled_verb_emb, weighted_code_embeddings_prep1, weighted_code_embeddings_prep2),
                                 dim=1)
        scores_out = self.model1.forward(model1_input)
        #         model2_input = torch.cat(
        #             (verb_embedding, prep1_embedding, prep2_embedding, scores_out.squeeze().unsqueeze(dim=0)),
        #             dim=1)
        model2_input = scores_out.squeeze().unsqueeze(dim=0)
        emb_out = self.model2.forward(model2_input)
        l1_reg_loss = torch.norm(scores_out, 1)
        return emb_out, scores_out, l1_reg_loss
