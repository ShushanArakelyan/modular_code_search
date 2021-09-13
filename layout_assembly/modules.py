import torch
from layout_assembly.action_v1 import ActionModule_v1_one_input, ActionModule_v1_two_inputs

from layout_assembly.utils import ProcessingException
from scoring.embedder import Embedder


class ActionModuleFacade_v1:
    def __init__(self, device, checkpoint=None, eval=False):
        self.device = device
        self.one_input_module = ActionModule_v1_one_input(device, eval)
        self.two_inputs_module = ActionModule_v1_two_inputs(device, eval)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}
        if checkpoint:
            self.load_from_checkpoint(checkpoint)

    def forward(self, verb, inputs, code):
        num_inputs = len(inputs)
        if num_inputs > 2:
            raise ProcessingException()
        module = self.modules[num_inputs]
        emb, pred = module.forward(verb, inputs, code)
        return emb, pred

    def parameters(self):
        return self.one_input_module.parameters() + self.two_inputs_module.parameters()

    def save_to_checkpoint(self, checkpoint):
        state_dict = {'one_input': self.one_input_module.state_dict(),
                      'two_inputs': self.two_inputs_module.state_dict()}
        torch.save(state_dict, checkpoint)

    def load_from_checkpoint(self, checkpoint):
        models = torch.load(checkpoint, map_location=self.device)
        self.one_input_module.load_state_dict(models['one_input'])
        self.two_inputs_module.load_state_dict(models['two_inputs'])


class ScoringModule:
    def __init__(self, device, checkpoint=None, eval=True):
        self.embedder = Embedder(device, model_eval=eval)
        self.scorer = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(device)
        if eval:
            self.scorer.eval()
        self.device = device
        if checkpoint:
            models = torch.load(checkpoint, map_location=device)
            self.scorer.load_state_dict(models['scorer'])
            self.scorer = self.scorer.to(device)
            self.embedder.model.load_state_dict(models['embedder'])
            self.embedder.model = self.embedder.model.to(device)

    def forward_(self, query, sample):
        with torch.no_grad():
            _, code, static_tags, regex_tags, ccg_parse = sample
            query_embedder_out = self.embedder.embed(query, [' '])
            if query_embedder_out is None:
                raise ProcessingException()
            word_token_id_mapping, word_token_embeddings, _, __, ___, ____, cls_token_embedding = query_embedder_out
            code_embedder_out = self.embedder.embed([' '], code)
            if code_embedder_out is None:
                raise ProcessingException()
            _, __, code_token_id_mapping, code_embedding, ___, truncated_code_tokens, ____ = code_embedder_out
            if word_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
                raise ProcessingException()

            tiled_emb = cls_token_embedding.repeat(len(truncated_code_tokens), 1)
            forward_input = torch.cat((tiled_emb, code_embedding), dim=1)
            token_count = max(code_token_id_mapping[-1])
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()[:token_count]
            return scorer_out

    def forward_v2(self, query, sample):
        with torch.no_grad():
            _, code, static_tags, regex_tags, ccg_parse = sample
            embedder_out = self.embedder.embed(query, code)
            if embedder_out is None:
                raise ProcessingException()
            word_token_id_mapping, word_token_embeddings, code_token_id_mapping, code_embedding, _, truncated_code_tokens, cls_token_embedding = embedder_out

            if word_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
                raise ProcessingException()

            tiled_emb = cls_token_embedding.repeat(len(truncated_code_tokens), 1)
            forward_input = torch.cat((tiled_emb, code_embedding), dim=1)
            token_count = max(code_token_id_mapping[-1])
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()[:token_count]
            # padding_size = self.embedder.max_seq_length - len(scorer_out)

            print(scorer_out.shape, token_count)
            return scorer_out

    #     def forward_batch(self, queries, sample):
    #         with torch.no_grad():
    #             scorer_out = []
    #             _, code, static_tags, regex_tags, ccg_parse = sample

    #             code_embedder_out = self.embedder.embed([' '], code)
    #             if code_embedder_out is None:
    #                 raise ProcessingException()
    #             _, __, self.code_token_id_mapping, self.code_embedding, ___, self.truncated_code_tokens, ____ = code_embedder_out
    #             if self.code_token_id_mapping.size == 0:
    #                 raise ProcessingException()
    #             token_count = max(self.code_token_id_mapping[-1])

    #             for query in queries:
    #                 query_embedder_out = self.embedder.embed(query, [' '])
    #                 if query_embedder_out is None:
    #                     scorer_out.append(None)
    #                 self.word_token_id_mapping, self.word_token_embeddings, _, __, ___, ____, self.cls_token_embedding = query_embedder_out

    #                 if self.word_token_id_mapping.size == 0:
    #                     scorer_out.append(None)

    #                 tiled_emb = self.cls_token_embedding.repeat(len(self.truncated_code_tokens), 1)
    #                 forward_input = torch.cat((tiled_emb, self.code_embedding), dim=1)
    #                 scorer_out.append(torch.sigmoid(self.scorer.forward(forward_input)).squeeze()[:token_count])
    #             return scorer_out
