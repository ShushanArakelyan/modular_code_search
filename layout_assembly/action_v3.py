import torch

from layout_assembly.action_v1 import ActionModule_v1
from layout_assembly.utils import ProcessingException
import roberta_embedder as embedder


def get_verb_embedding(verb):
    with torch.inference_mode:
        return embedder.forward(verb)


class ActionModule_v3_one_input(ActionModule_v1):
    def __init__(self, device, eval=False):
        ActionModule_v1.__init__(self, device)
        dim = embedder.dim
        self.model1 = torch.nn.Sequential(torch.nn.Linear(dim * 3 + 1, dim),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(dim, 1)).to(device)
        self.model2 = torch.nn.Sequential(torch.nn.Linear(dim * 2 + 1, dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(dim, dim)).to(self.device)  # outputs an embedding

        if eval:
            self.eval()

    def forward(self, verb, arg1, code_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            emb, scores = scores
            prep_embedding = (emb + prep_embedding) / 2
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)

        verb_embedding = get_verb_embedding(verb)

        tiled_verb_emb = verb_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep_emb = prep_embedding.repeat(embedder.max_seq_length, 1)
        model1_input = torch.cat((tiled_verb_emb, tiled_prep_emb, code_embeddings, scores), dim=1)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep_embedding, self.scores_out.squeeze().unsqueeze(dim=0)), dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out

    def parameters(self):
        return list(self.model1.parameters()) + list(self.model2.parameters())


class ActionModule_v3_two_inputs(ActionModule_v1):
    def __init__(self, device, eval=False):
        ActionModule_v1.__init__(self, device)
        dim = embedder.dim
        # outputs a sequence of scores
        self.model1 = torch.nn.Sequential(torch.nn.Linear(dim * 4 + 2, dim),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(dim, 1)).to(self.device)
        # outputs an embedding
        self.model2 = torch.nn.Sequential(torch.nn.Linear(dim * 3 + 1, dim), torch.nn.ReLU(),
                                          torch.nn.Linear(dim, dim)).to(self.device)
        if eval:
            self.eval()

    def forward(self, verb, args, code_embeddings):
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

        verb_embedding = get_verb_embedding(verb)

        tiled_verb_emb = verb_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep1_emb = prep1_embedding.repeat(embedder.max_seq_length, 1)
        tiled_prep2_emb = prep2_embedding.repeat(embedder.max_seq_length, 1)
        model1_input = torch.cat(
            (tiled_verb_emb, tiled_prep1_emb, tiled_prep2_emb, code_embeddings, scores1, scores2), dim=1)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep1_embedding, prep2_embedding, self.scores_out.squeeze().unsqueeze(dim=0)), dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out


# class ActionModule_v2_1_one_input(ActionModule_v2_one_input):
#     def __init__(self, device):
#         ActionModule_v2_one_input.__init__(self, device)
#         self.model1 = FC_Hypernetwork(self.model1, device)
#         self.model2 = FC_Hypernetwork(self.model2, device)
#
#     def set_hyper_param(self, verb_embedding):
#         ActionModule_v2_one_input.set_hyper_param(self, verb_embedding)
#         self.model2.set_hyper_param(verb_embedding)
#
#
# class ActionModule_v2_1_two_inputs(ActionModule_v2_two_inputs):
#     def __init__(self, device):
#         ActionModule_v2_two_inputs.__init__(self, device)
#         self.model1 = FC_Hypernetwork(self.model1, device)
#         self.model2 = FC_Hypernetwork(self.model2, device)
#
#     def embed_verb(self, verb_embedding):
#         ActionModule_v2_two_inputs.set_hyper_param(self, verb_embedding)
#         self.model2.set_hyper_param(verb_embedding)
