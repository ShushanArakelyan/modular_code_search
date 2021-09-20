from itertools import chain

import torch

import codebert_embedder as embedder
from layout_assembly.action_v1 import ActionModule_v1_one_input, ActionModule_v1_two_inputs
from layout_assembly.action_v2 import ActionModule_v2_one_input, ActionModule_v2_two_inputs
from layout_assembly.action_v3 import ActionModule_v3_one_input, ActionModule_v3_two_inputs
from layout_assembly.action_v4 import ActionModule_v4_one_input, ActionModule_v4_two_inputs
from layout_assembly.action_adapter import ActionModule_v1_reduced_one_input, ActionModule_v1_reduced_two_inputs
from layout_assembly.utils import ProcessingException


class ActionModuleFacade_v1:
    def __init__(self, device, checkpoint=None, eval=False):
        self.device = device
        self.one_input_module = ActionModule_v1_one_input(device, eval)
        self.two_inputs_module = ActionModule_v1_two_inputs(device, eval)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}
        if checkpoint:
            self.load_from_checkpoint(checkpoint)

    def forward(self, verb, inputs, code, verb_embedding):
        num_inputs = len(inputs)
        if num_inputs > 2:
            raise ProcessingException()
        module = self.modules[num_inputs]
        emb, pred = module.forward(verb, inputs, code, verb_embedding)
        return emb, pred

    def parameters(self):
        return chain(self.one_input_module.parameters(), self.two_inputs_module.parameters())

    def save_to_checkpoint(self, checkpoint):
        state_dict = {'one_input': self.one_input_module.state_dict(),
                      'two_inputs': self.two_inputs_module.state_dict()}
        torch.save(state_dict, checkpoint)

    def load_from_checkpoint(self, checkpoint):
        models = torch.load(checkpoint, map_location=self.device)
        self.one_input_module.load_state_dict(models['one_input'])
        self.two_inputs_module.load_state_dict(models['two_inputs'])


class ActionModuleFacade_v2(ActionModuleFacade_v1):
    def __init__(self, device, checkpoint=None, eval=False):
        ActionModuleFacade_v1.__init__(self, device)
        self.one_input_module = ActionModule_v2_one_input(device, eval)
        self.two_inputs_module = ActionModule_v2_two_inputs(device, eval)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}
        if checkpoint:
            self.load_from_checkpoint(checkpoint)
            
            
class ActionModuleFacade_v3(ActionModuleFacade_v1):
    def __init__(self, device, checkpoint=None, eval=False):
        ActionModuleFacade_v1.__init__(self, device)
        self.one_input_module = ActionModule_v3_one_input(device, eval)
        self.two_inputs_module = ActionModule_v3_two_inputs(device, eval)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}
        if checkpoint:
            self.load_from_checkpoint(checkpoint)

            
class ActionModuleFacade_v4(ActionModuleFacade_v1):
    def __init__(self, device, checkpoint=None, eval=False):
        ActionModuleFacade_v1.__init__(self, device)
        self.one_input_module = ActionModule_v4_one_input(device, eval)
        self.two_inputs_module = ActionModule_v4_two_inputs(device, eval)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}
        if checkpoint:
            self.load_from_checkpoint(checkpoint)

            
class ActionModuleFacade_v1_1_reduced(ActionModuleFacade_v1):
    def __init__(self, device, checkpoint=None, eval=False):
        ActionModuleFacade_v1.__init__(self, device)
        self.one_input_module = ActionModule_v1_reduced_one_input(device, eval)
        self.two_inputs_module = ActionModule_v1_reduced_two_inputs(device, eval)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}
        if checkpoint:
            self.load_from_checkpoint(checkpoint)


class ScoringModule:
    def __init__(self, device, checkpoint=None, eval=True):
        if not embedder.initialized:
            embedder.init_embedder(device)
        self.scorer = torch.nn.Sequential(torch.nn.Linear(embedder.dim * 2, embedder.dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(embedder.dim, 1)).to(device)
        if eval:
            self.scorer.eval()
        self.device = device
        if checkpoint:
            models = torch.load(checkpoint, map_location=device)
            self.scorer.load_state_dict(models['scorer'])
            self.scorer = self.scorer.to(device)
            embedder.model.load_state_dict(models['embedder'])
            embedder.model = embedder.model.to(device)

    def forward_batch(self, queries, codes, separate_embedding=False):
        if separate_embedding:
            raise Exception("Separate embedding not supported for processing in batch")
        with torch.no_grad():
            query_embeddings, code_embeddings = embedder.embed_batch(queries, codes)
            query_embeddings = query_embeddings.repeat(1, embedder.max_seq_length, 1)
            forward_input = torch.cat((query_embeddings, code_embeddings), dim=2)
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input))
        return scorer_out

    def forward(self, query, sample, separate_embedding=False):
        if separate_embedding:
            return self.forward_separate_emb(query, sample)
        else:
            return self.forward_combined_emb(query, sample)

    def forward_separate_emb(self, query, sample):
        with torch.no_grad():
            _, code, static_tags, regex_tags, ccg_parse = sample
            query_embedder_out = embedder.embed(query, [' '])
            if query_embedder_out is None:
                raise ProcessingException()
            word_token_id_mapping, word_token_embeddings, _, __, ___, ____, cls_token_embedding = query_embedder_out
            code_embedder_out = embedder.embed([' '], code)
            if code_embedder_out is None:
                raise ProcessingException()
            _, _, _, code_embeddings, _, _, _ = code_embedder_out
            padding_size = embedder.max_seq_length - len(code_embeddings)
            code_embeddings = torch.nn.functional.pad(code_embeddings, (0, 0, 0, padding_size), 'constant', 0)
            cls_token_embedding = cls_token_embedding.repeat(embedder.max_seq_length, 1)
            forward_input = torch.cat((cls_token_embedding, code_embeddings), dim=1)
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()
        return scorer_out

    def forward_combined_emb(self, query, code):
        with torch.no_grad():
            embedder_out = embedder.embed(query, code)
            if embedder_out is None:
                raise ProcessingException()
            word_token_id_mapping, _, code_token_id_mapping, code_embedding, _, _, cls_token_embedding = embedder_out

            if word_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
                raise ProcessingException()

            padding_size = embedder.max_seq_length - len(code_embedding)
            code_embedding = torch.nn.functional.pad(code_embedding, (0, 0, 0, padding_size), 'constant', 0)
            cls_token_embedding = cls_token_embedding.repeat(embedder.max_seq_length, 1)
            forward_input = torch.cat((cls_token_embedding, code_embedding), dim=1)
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()
            return scorer_out
