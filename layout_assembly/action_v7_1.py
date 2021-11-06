import torch
import torch.nn as nn

import codebert_embedder_v2 as embedder
from layout_assembly.action_v5 import ActionModule_v5
from layout_assembly.utils import ProcessingException, FC2, FC2_normalized, init_weights


class ActionModule_v7_1_one_input(ActionModule_v5):
    def init_networks(self):
        # outputs a sequence of scores
        self.input_dim = 2312
        self.padding_size = self.input_dim - (embedder.dim * 3 + 1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=8).to(self.device)
        hidden_input_dims = [self.input_dim, 512]
        hidden_output_dims = [512, 512]
        if self.normalized:
            self.mlp = FC2_normalized(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        else:
            self.mlp = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        self.mlp.apply(init_weights)

    def forward(self, _, arg1, __, precomputed_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            _, scores = scores
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)
        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        seq_len = code_embeddings.shape[0]  # tile verb and prep embeddings to the same shape as code
        tiled_verb_emb = verb_embedding.repeat(seq_len, 1)
        tiled_prep_emb = prep_embedding.repeat(seq_len, 1)
        tiled_cls_emb = (torch.ones((1, self.input_dim)) * (embedder.cls_value)).to(self.device)
        tiled_sep_emb = (torch.ones((1, self.input_dim)) * (embedder.sep_value)).to(self.device)
        padding_zeros = torch.zeros((seq_len, self.padding_size))
        code_embeddings = torch.cat((code_embeddings, padding_zeros), dim=1)
        encoder_input = torch.cat((tiled_verb_emb, tiled_prep_emb, scores[:seq_len], code_embeddings), dim=1)
        encoder_input = torch.cat((tiled_cls_emb, encoder_input, tiled_sep_emb), dim=0).unsqueeze(dim=1)
        print("encoder input before padding: ", encoder_input.shape)
        encoder_input = torch.nn.functional.pad(encoder_input, (0, 0, 0, 0, 0, embedder.max_seq_length - encoder_input.shape[0]),
                                                 'constant', 0)
        print("encoder input after padding: ", encoder_input.shape)
        mlp_input = self.encoder_layer(encoder_input).squeeze()
        print("mlp input: ", mlp_input.shape)
#         mlp_input = torch.index_select(mlp_input, 0, torch.LongTensor(range(1, seq_len)).to(self.device)).squeeze()
        mlp_input = torch.mm(scores.T, mlp_input)
        print("mlp input: ", mlp_input.shape)
        scores_out = self.mlp.forward(mlp_input).T
#         scores_out = torch.nn.functional.pad(scores_out, (0, 0, 0, embedder.max_seq_length - scores_out.shape[0]),
#                                              'constant', 0)
        print("scores_out: ", scores_out.shape)
        l1_reg_loss = torch.norm(scores_out, 1)
        return None, scores_out, l1_reg_loss


class ActionModule_v7_1_two_inputs(ActionModule_v5):
    def init_networks(self):
        # self.input_dim = embedder.dim * 4 + 2
        self.input_dim = 3080
        self.padding_size = self.input_dim - (embedder.dim * 4 + 2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=8).to(self.device)
        hidden_input_dims = [self.input_dim * 2, 512]
        hidden_output_dims = [512, 512]
        if self.normalized:
            self.mlp = FC2_normalized(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        else:
            self.mlp = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        self.mlp.apply(init_weights)

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
        seq_len = code_embeddings.shape[0]  # tile verb and prep embeddings to the same shape as code
        padding_zeros = torch.zeros((seq_len, self.padding_size))
        code_embeddings = torch.cat((code_embeddings, padding_zeros), dim=1)
        tiled_verb_emb = verb_embedding.repeat(seq_len, 1)
        tiled_prep1_emb = prep1_embedding.repeat(seq_len, 1)
        tiled_prep2_emb = prep2_embedding.repeat(seq_len, 1)
        tiled_cls_emb = (torch.ones((1, self.input_dim)) * (embedder.cls_value)).to(self.device)
        tiled_sep_emb = (torch.ones((1, self.input_dim)) * (embedder.sep_value)).to(self.device)

        encoder_input = torch.cat(
            (tiled_verb_emb, tiled_prep1_emb, scores1[:seq_len], tiled_prep2_emb, scores2[:seq_len], code_embeddings),
            dim=1)
        encoder_input = torch.cat((tiled_cls_emb, encoder_input, tiled_sep_emb), dim=0).unsqueeze(dim=1)
        print("encoder input before padding: ", encoder_input.shape)
        encoder_input = torch.nn.functional.pad(encoder_input, (0, 0, 0, 0, 0, embedder.max_seq_length - encoder_input.shape[0]),
                                             'constant', 0)
        print("encoder input after padding: ", encoder_input.shape)
        mlp_input = self.encoder_layer(encoder_input).squeeze()
        print("mlp input: ", mlp_input.shape)
        # mlp_input = torch.index_select(mlp_input, 0, torch.LongTensor(range(1, seq_len)).to(self.device)).squeeze()
        mlp_input_1 = torch.mm(scores1.T, mlp_input)
        mlp_input_2 = torch.mm(scores2.T, mlp_input)
        mlp_input = torch.cat((mlp_input_1, mlp_input_2), dim=1)
        print("mlp input: ", mlp_input.shape)
        scores_out = self.mlp.forward(mlp_input).T
        print("scores_out: ", scores_out.shape)

        l1_reg_loss = torch.norm(scores_out, 1)
        return None, scores_out, l1_reg_loss
