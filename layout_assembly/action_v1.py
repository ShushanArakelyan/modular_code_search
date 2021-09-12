import torch

from scoring.embedder import Embedder
from layout_assembly.utils import ProcessingException


class ActionModule_v1:
    def __init__(self, device):
        self.device = device
        self.embedder = Embedder(device, model_eval=True)
        self.model1 = None
        self.model2 = None
        self.scores_out = None
        self.emb_out = None
   
    def parameters(self):
        return list(self.model1.parameters()) + list(self.model2.parameters())
    
    def load_state_dict(self, d):
        self.model1.load_state_dict(d['model1'])
        self.model1 = self.model1.to(self.device)
        self.model2.load_state_dict(d['model2'])
        self.model2 = self.model2.to(self.device)
        
    def state_dict(self):
        return {'model1': self.model1.state_dict(), 'model2': self.model2.state_dict}
        

class ActionModule_v1_one_input(ActionModule_v1):
    def __init__(self, device):
        ActionModule_v1.__init__(self, device)
        self.model1 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 3 + 1, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(
            self.device)  # outputs a sequence of scores
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

        verb_embedding_out = self.embedder.embed([verb], [' '])
        code_embeddings_out = self.embedder.embed([' '], code_tokens)
        if verb_embedding_out is None or code_embeddings_out is None:
#             print('verb embedding out: ', verb_embedding_out)
#             print('code_embeddings_out: ', code_embeddings_out)
            raise ProcessingException()
        verb_embedding = verb_embedding_out[1]
        _, __, code_token_id_mapping, code_embeddings, _, truncated_code, ___ = code_embeddings_out
        token_count = max(code_token_id_mapping[-1])
        tiled_verb_emb = verb_embedding.repeat(token_count, 1)
        tiled_prep_emb = prep_embedding.repeat(token_count, 1)
        model1_input = torch.cat((tiled_verb_emb, tiled_prep_emb, code_embeddings[:token_count], scores), dim=1)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep_embedding), dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out


class ActionModule_v1_two_inputs(ActionModule_v1):
    def __init__(self, device):
        ActionModule_v1.__init__(self, device)
        self.model1 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 4 + 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(
            self.device)  # outputs a sequence of scores
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

        verb_embedding_out = self.embedder.embed([verb], [' '])
        code_embeddings_out = self.embedder.embed([' '], code_tokens)
        if verb_embedding_out is None or code_embeddings_out is None:
#             print('verb embedding out: ', verb_embedding_out)
#             print('code_embeddings_out: ', code_embeddings_out)
            raise ProcessingException()
        verb_embedding = verb_embedding_out[1]
        _, __, code_token_id_mapping, code_embeddings, _, truncated_code, ___ = code_embeddings_out

        token_count = max(code_token_id_mapping[-1])
        tiled_verb_emb = verb_embedding.repeat(token_count, 1)
        tiled_prep1_emb = prep1_embedding.repeat(token_count, 1)
        tiled_prep2_emb = prep2_embedding.repeat(token_count, 1)
        model1_input = torch.cat(
            (tiled_verb_emb, tiled_prep1_emb, tiled_prep2_emb, code_embeddings[:token_count], scores1, scores2), dim=1)
        self.scores_out = self.model1.forward(model1_input)
        print(self.scores_out.shape)
        model2_input = torch.cat((verb_embedding, prep1_embedding, prep2_embedding), dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out
