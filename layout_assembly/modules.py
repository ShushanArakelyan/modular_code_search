import torch

from scoring.embedder import Embedder


class ActionModule_v1_one_input:
    def __init__(self, device):
        self.device = device
        self.embedder = Embedder(device, model_eval=True)
        self.model1 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 3 + 1, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(self.device) # outputs a sequence of scores
        self.model2 = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), self.embedder.get_dim())).to(self.device) # outputs an embedding
        self.scores_out = None
        self.emb_out = None
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.op = torch.optim.Adam(list(self.model1.parameters()) + list(self.model1.model.parameters()), lr=1e-5)

    def forward(self, verb, arg1, code_tokens):
        prep, scores = arg1

        verb_embedding_out = self.embedder.embed([verb], [' '])
        prep_embedding_out = self.embedder.embed([prep], [' '])
        code_embeddings_out = self.embedder.embed([' '], code_tokens)
        if verb_embedding_out is None or prep_embedding_out is None or code_embeddings_out is None:
            print('verb embedding out: ', verb_embedding_out)
            print('prep_embedding_out: ', prep_embedding_out)
            print('code_embeddings_out: ', code_embeddings_out)
            return None
        verb_embedding = verb_embedding_out[1]
        prep_embedding = prep_embedding_out[1]
        code_embeddings, truncated_code = code_embeddings_out[3],  code_embeddings_out[5]
        tiled_verb_emb = verb_embedding.repeat(len(truncated_code), 1)
        tiled_prep_emb = prep_embedding.repeat(len(truncated_code), 1)
        model1_input = torch.cat((tiled_verb_emb, tiled_prep_emb, code_embeddings, scores), dim=1)
        print('model1 input.shape: ', model1_input.shape)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep_embedding), dim=1)
        print('model2 input.shape: ', model2_input.shape)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out

    def backward(self, y):
        loss = self.loss_func(y, self.emb_out)
        loss.backward()
        self.op.step()
        return loss.data


class ActionModule_v1_two_inputs:
    def __init__(self, device):
        self.device = device
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
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.op = torch.optim.Adam(list(self.model1.parameters()) + list(self.model1.model.parameters()), lr=1e-5)

    def forward(self, verb, args, code_tokens):
        arg1, arg2 = args
        prep1, scores1 = arg1
        prep2, scores2 = arg2

        verb_embedding_out = self.embedder.embed([verb], [' '])
        prep1_embedding_out = self.embedder.embed([prep1], [' '])
        prep2_embedding_out = self.embedder.embed([prep2], [' '])
        code_embeddings_out = self.embedder.embed([' '], code_tokens)
        if verb_embedding_out is None or prep1_embedding_out is None or prep2_embedding_out is None or code_embeddings_out is None:
            print('verb embedding out: ', verb_embedding_out)
            print('prep1_embedding_out: ', prep1_embedding_out)
            print('prep2_embedding_out: ', prep2_embedding_out)
            print('code_embeddings_out: ', code_embeddings_out)
            return None
        verb_embedding = verb_embedding_out[1]
        prep1_embedding = prep1_embedding_out[1]
        prep2_embedding = prep2_embedding_out[1]
        code_embeddings, truncated_code = code_embeddings_out[3], code_embeddings_out[5]
        tiled_verb_emb = verb_embedding.repeat(len(truncated_code), 1)
        tiled_prep1_emb = prep1_embedding.repeat(len(truncated_code), 1)
        tiled_prep2_emb = prep2_embedding.repeat(len(truncated_code), 1)
        model1_input = torch.cat((tiled_verb_emb, tiled_prep1_emb, tiled_prep2_emb, code_embeddings, scores1, scores2), dim=1)
        print('model1 input.shape: ', model1_input.shape)
        self.scores_out = self.model1.forward(model1_input)
        model2_input = torch.cat((verb_embedding, prep1_embedding, prep2_embedding), dim=1)
        print('model2 input.shape: ', model2_input.shape)
        self.emb_out = self.model2.forward(model2_input)
        return self.emb_out, self.scores_out

    def backward(self, y):
        loss = self.loss_func(y, self.emb_out)
        loss.backward()
        self.op.step()
        return loss.data


class ActionModuleFacade_v1:
    def __init__(self):
        self.one_input_module = ActionModule_v1_one_input()
        self.two_inputs_module = ActionModule_v1_two_inputs()
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}

    def forward(self, verb, inputs, code):
        num_inputs = len(inputs)
        assert num_inputs <= 2, f'Too many inputs, handling {num_inputs} inputs is not implemented for ActionModule_v1'
        module = self.modules[num_inputs]
        return module.forward(verb, inputs, code)


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

    def forward(self, sample):
        query, code, static_tags, regex_tags, ccg_parse = sample
        embedder_out = self.embedder.embed(query, code)
        if embedder_out is None:
            return None
        self.word_token_id_mapping, self.word_token_embeddings, self.code_token_id_mapping, \
        self.code_embedding, _, self.truncated_code_tokens, self.cls_token_embedding = embedder_out
        if self.word_token_id_mapping.size == 0 or self.code_token_id_mapping.size == 0:
            return None

        tiled_emb = self.cls_token_embedding.repeat(len(self.truncated_code_tokens), 1)
        forward_input = torch.cat((tiled_emb, self.code_embedding), dim=1)
        token_count = max(self.code_token_id_mapping[-1])
        scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze().cpu().detach().numpy()[:token_count]
        return scorer_out

    def compute_loss(self):
        pass
