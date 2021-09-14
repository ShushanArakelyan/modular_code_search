import torch

from layout_assembly.utils import ProcessingException
from scoring.embedder import Embedder


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
        return {'model1': self.model1.state_dict(), 'model2': self.model2.state_dict()}

    def eval(self):
        self.model1.eval()
        self.model2.eval()

    def train(self):
        self.model1.train()
        self.model2.train()

    def embed_verb(self, verb):
        verb_embedding_out = self.embedder.embed([verb], [' '])
        if verb_embedding_out is None:
            raise ProcessingException()
        verb_embedding = verb_embedding_out[1]
        return verb_embedding

    def embed_code(self, code):
        code_embeddings_out = self.embedder.embed([' '], code, fast=True)
        if code_embeddings_out is None:
            raise ProcessingException()
        _, _, _, code_embeddings, _, _, _ = code_embeddings_out
        padding_size = self.embedder.max_seq_length - len(code_embeddings)
        code_embeddings = torch.nn.functional.pad(code_embeddings, (0, 0, 0, padding_size), 'constant', 0)
        return code_embeddings


class ActionModule_v1_one_input(ActionModule_v1):
    def __init__(self, device, eval=False):
        ActionModule_v1.__init__(self, device)
        self.model1 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 3 + 1, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(
            self.device)  # outputs a sequence of scores
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(self.embedder.get_dim() * 2 + self.embedder.max_seq_length, self.embedder.get_dim()),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedder.get_dim(), self.embedder.get_dim())).to(
            self.device)  # outputs an embedding
        if eval:
            self.eval()

    def forward(self, verb, arg1, code_tokens):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            emb, scores = scores
            prep_embedding = (emb + prep_embedding) / 2
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)

        embedding_out = self.embedder.embed(verb, code_tokens, fast=True)
        if embedding_out is None:
            raise ProcessingException()
        _, _, _, code_embeddings, _, _, cls_token_embedding  = embedding_out
        
        padding_size = self.embedder.max_seq_length - len(code_embeddings)
        code_embeddings = torch.nn.functional.pad(code_embeddings, (0, 0, 0, padding_size), 'constant', 0)
        verb_embedding = cls_token_embedding

        tiled_verb_emb = verb_embedding.repeat(self.embedder.max_seq_length, 1)
        tiled_prep_emb = prep_embedding.repeat(self.embedder.max_seq_length, 1)
        model1_input = torch.cat((tiled_verb_emb, tiled_prep_emb, code_embeddings, scores), dim=1)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep_embedding, self.scores_out.squeeze().unsqueeze(dim=0)), dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out


class ActionModule_v1_two_inputs(ActionModule_v1):
    def __init__(self, device, eval=False):
        ActionModule_v1.__init__(self, device)
        self.model1 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 4 + 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(
            self.device)  # outputs a sequence of scores
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(self.embedder.get_dim() * 3 + self.embedder.max_seq_length, self.embedder.get_dim()),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedder.get_dim(), self.embedder.get_dim())).to(
            self.device)  # outputs an embedding
        if eval:
            self.eval()

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

        embedding_out = self.embedder.embed([verb], code_tokens, fast=True)
        if embedding_out is None:
            raise ProcessingException()
        _, _, _, code_embeddings, _, _, cls_token_embedding  = embedding_out
        
        padding_size = self.embedder.max_seq_length - len(code_embeddings)
        code_embeddings = torch.nn.functional.pad(code_embeddings, (0, 0, 0, padding_size), 'constant', 0)
        verb_embedding = cls_token_embedding
        tiled_verb_emb = verb_embedding.repeat(self.embedder.max_seq_length, 1)
        tiled_prep1_emb = prep1_embedding.repeat(self.embedder.max_seq_length, 1)
        tiled_prep2_emb = prep2_embedding.repeat(self.embedder.max_seq_length, 1)
        model1_input = torch.cat(
            (tiled_verb_emb, tiled_prep1_emb, tiled_prep2_emb, code_embeddings, scores1, scores2), dim=1)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep1_embedding, prep2_embedding, self.scores_out.squeeze().unsqueeze(dim=0)),
                                 dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out
