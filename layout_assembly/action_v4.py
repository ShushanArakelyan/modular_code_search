import torch

import codebert_embedder as embedder
from hypernetwork.hypernetwork import FC_Hypernetwork
from layout_assembly.action_v2 import ActionModule_v2
from layout_assembly.utils import ProcessingException


class ActionModule_v4(ActionModule_v2):
    def __init__(self, device, dropout=0):
        super(ActionModule_v4, self).__init__(device)
        self.reduce_dim_model = None
        self.dropout = dropout

    def parameters(self):
        return chain(self.model1.parameters(), self.model2.parameters(), self.reduce_dim_model.parameters())


# Hypernetwork, where the MLPs are parametrized by the verb
class ActionModule_v4_one_input(ActionModule_v2):
    def __init__(self, device):
        ActionModule_v2.__init__(self, device)
        dim = embedder.dim
        self.reduce_dim_model = torch.nn.Linear(dim, 32)
        self.model1 = FC_Hypernetwork(dim,
                                      torch.nn.Sequential(torch.nn.Linear(dim + 32 + 1, 128),
                                                          torch.nn.Dropout(self.dropout),
                                                          torch.nn.ReLU(),
                                                          torch.nn.Linear(128, 1)).to(device),
                                      device)
        self.model2 = torch.nn.Sequential(torch.nn.Linear(32 + embedder.max_seq_length, 128),
                                          torch.nn.Dropout(self.dropout),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, dim)).to(self.device)  # outputs an embedding

    def forward(self, verb, arg1, code_tokens, precomputed_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            prep_embedding = (scores[0] + prep_embedding) / 2
            prep_embedding = self.reduce_dim_model(prep_embedding)
            scores = scores[1]
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)

        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        self.set_hyper_param(verb_embedding)
        scores_out = self.model1.forward(torch.cat(
            (prep_embedding.repeat(embedder.max_seq_length, 1),
             code_embeddings,
             scores), dim=1))
        emb_out = self.model2.forward(
            torch.cat((verb_embedding, prep_embedding, scores_out.squeeze().unsqueeze(dim=0)), dim=1))
        return emb_out, scores_out


class ActionModule_v4_two_inputs(ActionModule_v2):
    def __init__(self, device):
        ActionModule_v2.__init__(self, device)
        dim = embedder.dim
        self.reduce_dim_model = torch.nn.Linear(dim, 32)
        # outputs a sequence of scores
        self.model1 = FC_Hypernetwork(dim,
                                      torch.nn.Sequential(torch.nn.Linear(dim + 2 * 32 + 2, 128),
                                                          torch.nn.Dropout(self.dropout),
                                                          torch.nn.ReLU(),
                                                          torch.nn.Linear(128, 1)).to(self.device),
                                      device)
        # outputs an embedding
        self.model2 = torch.nn.Sequential(torch.nn.Linear(2 * 32 + embedder.max_seq_length, 128),
                                          torch.nn.Dropout(self.dropout),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, dim)).to(self.device)

    def forward(self, verb, args, code_tokens, precomputed_embeddings):
        arg1, arg2 = args
        prep1_embedding, scores1 = arg1
        if isinstance(scores1, tuple):
            prep1_embedding = (scores1[0] + prep1_embedding) / 2
            prep1_embedding = self.reduce_dim_model(prep1_embedding)
            scores1 = scores1[1]
        if len(scores1.shape) == 1:
            scores1 = scores1.unsqueeze(dim=1)
        prep2_embedding, scores2 = arg2
        if isinstance(scores2, tuple):
            prep2_embedding = (scores2[0] + prep2_embedding) / 2
            prep2_embedding = self.reduce_dim_model(prep2_embedding)
            scores2 = scores2[1]
        if len(scores2.shape) == 1:
            scores2 = scores2.unsqueeze(dim=1)
        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        self.set_hyper_param(verb_embedding)
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

# class ActionModule_v2_1_one_input(ActionModule_v2_one_input):
#     def __init__(self, device):
#         ActionModule_v2_one_input.__init__(self, device)
#         self.model1 = FC_Hypernetwork(embedder.dim, self.model1, device)
#         self.model2 = FC_Hypernetwork(embedder.dim, self.model2, device)

#     def set_hyper_param(self, verb_embedding):
#         ActionModule_v2_one_input.set_hyper_param(self, verb_embedding)
#         self.model2.set_hyper_param(verb_embedding)


# class ActionModule_v2_1_two_inputs(ActionModule_v2_two_inputs):
#     def __init__(self, device):
#         ActionModule_v2_two_inputs.__init__(self, device)
#         self.model1 = FC_Hypernetwork(embedder.dim, self.model1, device)
#         self.model2 = FC_Hypernetwork(embedder.dim, self.model2, device)

#     def set_hyper_param(self, verb_embedding):
#         ActionModule_v2_1_two_inputs.set_hyper_param(self, verb_embedding)
#         self.model2.set_hyper_param(verb_embedding)
