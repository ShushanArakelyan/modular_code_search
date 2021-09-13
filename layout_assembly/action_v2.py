import torch

from hypernetwork.hypernetwork import myLinear, FC_Hypernetwork
from layout_assembly.action_v1 import ActionModule_v1


# Hypernetwork, where the MLPs are parametrized by the verb
class ActionModule_v2(ActionModule_v1):
    def embed_verb(self, verb):
        verb_embedding = ActionModule_v1.embed_verb(self, verb)
        self.model1.set_hyper_param(verb_embedding)
        return verb_embedding


class ActionModule_v2_one_input(ActionModule_v2):
    def __init__(self, device):
        ActionModule_v2.__init__(self, device)
        model1 = torch.nn.Sequential(myLinear(self.embedder.get_dim() * 3 + 1, self.embedder.get_dim()),
                                     torch.nn.ReLU(),
                                     myLinear(self.embedder.get_dim(), 1)).to(
            self.device)  # outputs a sequence of scores
        self.model1 = FC_Hypernetwork(model1, device)
        self.model2 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), self.embedder.get_dim())).to(
            self.device)  # outputs an embedding

    def forward(self, verb, arg1, code_tokens):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            emb, scores = scores
            prep_embedding = (emb + prep_embedding) / 2
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)

        verb_embedding = self.embed_verb(verb)
        code_embeddings, code_token_id_mapping, token_count = self.embed_code(code_tokens)

        tiled_verb_emb = verb_embedding.repeat(token_count, 1)
        tiled_prep_emb = prep_embedding.repeat(token_count, 1)
        model1_input = torch.cat((tiled_verb_emb, tiled_prep_emb, code_embeddings[:token_count], scores), dim=1)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep_embedding), dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out

    def parameters(self):
        return list(self.model1.parameters()) + list(self.model2.parameters())


class ActionModule_v2_two_inputs(ActionModule_v2):
    def __init__(self, device):
        ActionModule_v2.__init__(self, device)
        model1 = torch.nn.Sequential(myLinear(self.embedder.get_dim() * 4 + 2, self.embedder.get_dim()),
                                     torch.nn.ReLU(),
                                     myLinear(self.embedder.get_dim(), 1)).to(
            self.device)  # outputs a sequence of scores
        self.model1 = FC_Hypernetwork(model1, device)
        self.model2 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 3, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), self.embedder.get_dim())).to(
            self.device)  # outputs an embedding

    def forward(self, verb, args, code_tokens):
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

        verb_embedding = self.embed_verb(verb)
        code_embeddings, code_token_id_mapping, token_count = self.embed_code(code_tokens)

        tiled_verb_emb = verb_embedding.repeat(token_count, 1)
        tiled_prep1_emb = prep1_embedding.repeat(token_count, 1)
        tiled_prep2_emb = prep2_embedding.repeat(token_count, 1)
        model1_input = torch.cat(
            (tiled_verb_emb, tiled_prep1_emb, tiled_prep2_emb, code_embeddings[:token_count], scores1, scores2), dim=1)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep1_embedding, prep2_embedding), dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out


class ActionModule_v2_1_one_input(ActionModule_v2_one_input):
    def __init__(self, device):
        ActionModule_v2_one_input.__init__(self, device)
        self.model1 = FC_Hypernetwork(self.model1, device)
        self.model2 = FC_Hypernetwork(self.model2, device)

    def embed_verb(self, verb):
        verb_embedding = ActionModule_v2_one_input.embed_verb(self, verb)
        self.model2.set_hyper_param(verb_embedding)
        return verb_embedding


class ActionModule_v2_1_two_inputs(ActionModule_v2_two_inputs):
    def __init__(self, device):
        ActionModule_v2_two_inputs.__init__(self, device)
        self.model1 = FC_Hypernetwork(self.model1, device)
        self.model2 = FC_Hypernetwork(self.model2, device)

    def embed_verb(self, verb):
        verb_embedding = ActionModule_v2_two_inputs.embed_verb(self, verb)
        self.model2.set_hyper_param(verb_embedding)
        return verb_embedding
