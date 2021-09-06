import torch

from scoring.embedder import Embedder
from layout_assembly.utils import ProcessingException

class ActionModule_v1_one_input:
    def __init__(self, device):
        self.device = device
        self.embedder = Embedder(device, model_eval=True)
        self.model1 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 3 + 1, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(
            self.device)  # outputs a sequence of scores
        self.model2 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), self.embedder.get_dim())).to(
            self.device)  # outputs an embedding
        self.scores_out = None
        self.emb_out = None

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
            print('verb embedding out: ', verb_embedding_out)
            print('code_embeddings_out: ', code_embeddings_out)
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

    def parameters(self):
        return list(self.model1.parameters()) + list(self.model2.parameters())


class ActionModule_v1_two_inputs:
    def __init__(self, device):
        self.device = device
        self.embedder = Embedder(device, model_eval=True)
        self.model1 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 4 + 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(
            self.device)  # outputs a sequence of scores
        self.model2 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 3, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), self.embedder.get_dim())).to(
            self.device)  # outputs an embedding
        self.scores_out = None
        self.emb_out = None

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
            print('verb embedding out: ', verb_embedding_out)
            print('code_embeddings_out: ', code_embeddings_out)
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
        model2_input = torch.cat((verb_embedding, prep1_embedding, prep2_embedding), dim=1)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out

    def parameters(self):
        return list(self.model1.parameters()) + list(self.model2.parameters())


class ActionModuleFacade_v1:
    def __init__(self, device):
        self.device = device
        self.one_input_module = ActionModule_v1_one_input(device)
        self.two_inputs_module = ActionModule_v1_two_inputs(device)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}

    def forward(self, verb, inputs, code):
        self.num_inputs = len(inputs)
        if self.num_inputs > 2:
            raise ProcessingException()
        #         assert num_inputs <= 2, f'Too many inputs, handling {num_inputs} inputs is not implemented for ActionModule_v1'
        module = self.modules[self.num_inputs]
        self.emb, self.pred = module.forward(verb, inputs, code)
        return self.emb, self.pred

    def parameters(self):
        return self.one_input_module.parameters() + self.two_inputs_module.parameters()


class ScoringModule:
    def __init__(self, device, checkpoint=None):
        self.embedder = Embedder(device, model_eval=False)
        self.scorer = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(device)

        # TODO: not sure about these
        # self.op = torch.optim.Adam(list(self.scorer.parameters()) + list(self.embedder.model.parameters()), lr=1e-8)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

        self.device = device
        if checkpoint:
            models = torch.load(checkpoint, map_location=device)
            self.scorer.load_state_dict(models['scorer'])
            self.scorer = self.scorer.to(device)
            self.embedder.model.load_state_dict(models['embedder'])
            self.embedder.model = self.embedder.model.to(device)
            # self.op.load_state_dict(models['optimizer'])

    def forward(self, query, sample):
        _, code, static_tags, regex_tags, ccg_parse = sample
        query_embedder_out = self.embedder.embed(query, [' '])
        if query_embedder_out is None:
            raise ProcessingException()
        self.word_token_id_mapping, self.word_token_embeddings, _, __, ___, ____, self.cls_token_embedding = query_embedder_out
        code_embedder_out = self.embedder.embed([' '], code)
        if code_embedder_out is None:
            raise ProcessingException()
        _, __, self.code_token_id_mapping, self.code_embedding, ___, self.truncated_code_tokens, ____ = code_embedder_out
        if self.word_token_id_mapping.size == 0 or self.code_token_id_mapping.size == 0:
            raise ProcessingException()

        tiled_emb = self.cls_token_embedding.repeat(len(self.truncated_code_tokens), 1)
        forward_input = torch.cat((tiled_emb, self.code_embedding), dim=1)
        token_count = max(self.code_token_id_mapping[-1])
        #         print('in scoring: ', token_count, ', query: ', query)
        scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()[:token_count]
        return scorer_out

    def compute_loss(self):
        pass
