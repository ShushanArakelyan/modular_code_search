from itertools import chain

import torch
import torch.nn as nn

import codebert_embedder_v2 as embedder
from layout_assembly.utils import ProcessingException, FC2, FC2_normalized, init_weights
from layout_assembly.action_v8 import ActionModule_v8_one_input, ActionModule_v8_two_inputs


class ActionModule_v8_one_input_scoring_ablation(ActionModule_v8_one_input):
    def forward(self, _, arg1, __, precomputed_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            _, scores = scores
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)
        scores = torch.ones_like(scores).to(self.device) * 0.5
        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        # here the code embeddings have 1 more sep symbol at the beginning
        encoder_input = torch.cat((embedder.cls_embedding, verb_embedding, prep_embedding, code_embeddings)).unsqueeze(dim=1)
        mlp_input = self.encoder_layer(encoder_input)
        mlp_input = torch.index_select(mlp_input, 0,
                                       torch.LongTensor(range(4, len(encoder_input))).to(self.device)).squeeze()
        mlp_input = torch.mm(scores.T, mlp_input)
        scores_out = self.mlp.forward(mlp_input).T
        l1_reg_loss = torch.norm(scores_out, 1)
        return None, scores_out, l1_reg_loss


class ActionModule_v8_two_inputs_scoring_ablation(ActionModule_v8_two_inputs):
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
        # replace all scores with 0.5
        scores1 = torch.ones_like(scores1).to(self.device) * 0.5
        scores2 = torch.ones_like(scores2).to(self.device) * 0.5
        if precomputed_embeddings is None:
            raise ProcessingException()

        verb_embedding, code_embeddings = precomputed_embeddings
        encoder_input = torch.cat((embedder.cls_embedding, verb_embedding, prep1_embedding,
                                   prep2_embedding, code_embeddings)).unsqueeze(dim=1)
        mlp_input = self.encoder_layer(encoder_input)
        mlp_input = torch.index_select(mlp_input, 0,
                                       torch.LongTensor(range(5, len(encoder_input))).to(self.device)).squeeze()
        mlp_input_1 = torch.mm(scores1.T, mlp_input)
        mlp_input_2 = torch.mm(scores2.T, mlp_input)
        mlp_input = torch.cat((mlp_input_1, mlp_input_2), dim=1)
        scores_out = self.mlp.forward(mlp_input).T
        l1_reg_loss = torch.norm(scores_out, 1)
        return None, scores_out, l1_reg_loss
