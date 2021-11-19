from itertools import chain

import torch

import scoring as embedder  # this instance of embedder is different than in the rest of the network, it remains frozen
from action.action_adapter_v2 import ActionModule_v2_1_one_input
from action.action_v1 import ActionModule_v1_one_input, ActionModule_v1_two_inputs
from action.action_v1_weighted_sum import ActionModule_v1_one_input as ActionModule_v11_weighted_one_input
from action.action_v1_weighted_sum import ActionModule_v1_two_inputs as ActionModule_v11_weighted_two_inputs
from action.action_v2 import ActionModule_v2_one_input, ActionModule_v2_two_inputs
from action.action_v3 import ActionModule_v3_one_input, ActionModule_v3_two_inputs
from action.action_v5 import ActionModule_v5_one_input, ActionModule_v5_two_inputs
from action.action_v5_1 import ActionModule_v5_1_one_input, ActionModule_v5_1_two_inputs
from action.action_v6 import ActionModule_v6_one_input, ActionModule_v6_two_inputs
from action.action_v7 import ActionModule_v7_one_input, ActionModule_v7_two_inputs
from action.action_v7_1 import ActionModule_v7_1_one_input, ActionModule_v7_1_two_inputs
from action.action_v7_2 import ActionModule_v7_2_one_input, ActionModule_v7_2_two_inputs
from action.action_v8 import ActionModule_v8_one_input, ActionModule_v8_two_inputs
from action.action_v8_scoring_ablation import ActionModule_v8_one_input_scoring_ablation, ActionModule_v8_two_inputs_scoring_ablation
from action.action_v8_action_ablation import ActionModule_v8_one_input_action_ablation, ActionModule_v8_two_inputs_action_ablation
from action.action_v8_verb_ablation import ActionModule_v8_one_input_verb_ablation, ActionModule_v8_two_inputs_verb_ablation
from action.action_v8_preposition_ablation import ActionModule_v8_one_input_preposition_ablation, ActionModule_v8_two_inputs_preposition_ablation
from layout_assembly.utils import ProcessingException


class ActionModuleFacade:
    def __init__(self, device, version, normalized, dropout=0):
        self.device = device
        self.dropout=dropout
        self.one_input_module = None
        self.two_inputs_module = None
        self.three_inputs_module = None
        self.four_inputs_module = None
        self.init_networks(version, normalized)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}

    def forward(self, verb, inputs, code, verb_embedding):
        num_inputs = len(inputs)
        if num_inputs > 2:
            raise ProcessingException()
        module = self.modules[num_inputs]
        emb, pred, reg_loss = module.forward(verb, inputs, code, verb_embedding)
        return emb, pred, reg_loss

    def init_networks(self, version, normalized):
        if version == 1:
            self.one_input_module = ActionModule_v1_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v1_two_inputs(self.device, normalized, self.dropout)
        elif version == 2:
            self.one_input_module = ActionModule_v2_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v2_two_inputs(self.device, normalized, self.dropout)
        elif version == 3:
            self.one_input_module = ActionModule_v3_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v3_two_inputs(self.device, normalized, self.dropout)
        elif version == 5:
            self.one_input_module = ActionModule_v5_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v5_two_inputs(self.device, normalized, self.dropout)
        elif version == 51:
            self.one_input_module = ActionModule_v5_1_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v5_1_two_inputs(self.device, normalized, self.dropout)
        elif version == 6:
            self.one_input_module = ActionModule_v6_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v6_two_inputs(self.device, normalized, self.dropout)
        elif version == 7:
            self.one_input_module = ActionModule_v7_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v7_two_inputs(self.device, normalized, self.dropout)
        elif version == 71:
            self.one_input_module = ActionModule_v7_1_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v7_1_two_inputs(self.device, normalized, self.dropout)
        elif version == 72:
            self.one_input_module = ActionModule_v7_2_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v7_2_two_inputs(self.device, normalized, self.dropout)
        elif version == 8:
            self.one_input_module = ActionModule_v8_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v8_two_inputs(self.device, normalized, self.dropout)
        elif version == 81:
            print("Scoring ablation")
            self.one_input_module = ActionModule_v8_one_input_scoring_ablation(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v8_two_inputs_scoring_ablation(self.device, normalized, self.dropout)
        elif version == 82:
            self.one_input_module = ActionModule_v8_one_input_action_ablation(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v8_two_inputs_action_ablation(self.device, normalized, self.dropout)
        elif version == 83:
            self.one_input_module = ActionModule_v8_one_input_verb_ablation(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v8_two_inputs_verb_ablation(self.device, normalized, self.dropout)
        elif version == 84:
            self.one_input_module = ActionModule_v8_one_input_preposition_ablation(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v8_two_inputs_preposition_ablation(self.device, normalized, self.dropout)
        elif version == 11:
            self.one_input_module = ActionModule_v11_weighted_one_input(self.device, normalized, self.dropout)
            self.two_inputs_module = ActionModule_v11_weighted_two_inputs(self.device, normalized, self.dropout)
#         elif version == 11:
#             raise Exception('This code has not been refactored')
#             self.one_input_module = ActionModule_v1_1_one_input(self.device, self.dropout)
#             self.two_inputs_module = ActionModule_v1_1_two_inputs(self.device, self.dropout)
        elif version == 21:
            raise Exception('This code has not been refactored')
            self.one_input_module = ActionModule_v2_1_one_input(self.device, self.dropout)
            self.two_inputs_module = ActionModule_v2_1_one_input(self.device, self.dropout)
        else:
            raise Exception("Unknown Action version")

    def parameters(self):
        return chain(self.one_input_module.parameters(), self.two_inputs_module.parameters())

    def named_parameters(self):
        return chain(self.one_input_module.named_parameters(), self.two_inputs_module.named_parameters())
    
    def state_dict(self):
        state_dict = {'one_input': self.one_input_module.state_dict(),
                      'two_inputs': self.two_inputs_module.state_dict()}
        return state_dict

    def save_to_checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def load_from_checkpoint(self, checkpoint):
        models = torch.load(checkpoint, map_location=self.device)
        self.one_input_module.load_state_dict(models['one_input'])
        self.two_inputs_module.load_state_dict(models['two_inputs'])
        
    def set_eval(self):
        self.one_input_module.eval()
        self.two_inputs_module.eval()
    
    def set_train(self):
        self.one_input_module.train()
        self.two_inputs_module.train()


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
            # query_embeddings = query_embeddings.repeat(1, embedder.max_seq_length, 1)
            assert len(query_embeddings) == len(code_embeddings)
            scorer_out = []
            for qe, ce in zip(query_embeddings, code_embeddings):
                # forward_input = torch.cat((query_embeddings, code_embeddings), dim=2)
                print(qe.shape, ce.shape)
                scorer_out.append(torch.sigmoid(self.scorer.forward(torch.cat((qe, ce), dim=1).unsqueeze(dim=0))))
        scorer_out = torch.cat(scorer_out, dim=0)
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

            # padding_size = embedder.max_seq_length - len(code_embedding)
            # code_embedding = torch.nn.functional.pad(code_embedding, (0, 0, 0, padding_size), 'constant', 0)
            cls_token_embedding = cls_token_embedding.repeat(embedder.max_seq_length, 1)
            forward_input = torch.cat((cls_token_embedding, code_embedding), dim=1)
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()
            return scorer_out
