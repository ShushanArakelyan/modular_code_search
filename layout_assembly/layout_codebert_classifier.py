from itertools import chain

import torch

import codebert_embedder_v2 as embedder
from layout_assembly.utils import ProcessingException, init_weights
from layout_assembly.layout import LayoutNet, LayoutNode


class ActionModuleWrapper(object):
    empty_emb = None
    prep_emb_cache = {}

    def __init__(self, action_module_facade):
        self.param = None
        self.inputs = []
        self.input_count = 0
        self.module = action_module_facade

    def forward(self, code, verb_embedding):
        return self.module.forward(self.param, self.inputs, code, verb_embedding)

    def add_input(self, input):
        if len(self.inputs) > 0 and len(self.inputs[-1]) == 1:
            self.inputs[-1].append(input)
        else:
            if ActionModuleWrapper.empty_emb is None:
                with torch.no_grad():
                    ActionModuleWrapper.empty_emb = embedder.embed([' '], [' '])[1]
            self.inputs.append([ActionModuleWrapper.empty_emb, input])

    def add_preposition(self, prep):
        if prep not in ActionModuleWrapper.prep_emb_cache:
            with torch.no_grad():
                ActionModuleWrapper.prep_emb_cache[prep] = embedder.embed([prep], [' '])[1]
        self.inputs.append([ActionModuleWrapper.prep_emb_cache[prep]])


class LayoutNet_w_codebert_classifier(LayoutNet):
    def __init__(self, scoring_module, action_module_facade, device, return_separators=False, precomputed_scores_provided=False, finetune_codebert=True):
        print(device)
        self.scoring_module = scoring_module
        self.action_module_facade = action_module_facade
        self.device = device
        self.precomputed_scores_provided = precomputed_scores_provided
        self.return_separators=return_separators
        dim = embedder.dim
        half_dim = int(dim / 2)

        self.classifier = embedder.classifier
        self.scoring_outputs = None
        self.accumulated_loss = None
    def parameters(self):
        return chain(self.classifier.parameters(), self.action_module_facade.parameters())

    def forward(self, ccg_parse, sample):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        code = sample[1]
        if len(code) == 0:  # erroneous example
            return None
        try:
            scoring_inputs, verb_embeddings = self.precompute_inputs(tree, code, [[], [], []], [[], []], '')
            if not self.precomputed_scores_provided:
                self.scoring_outputs = self.scoring_module.forward_batch(scoring_inputs[0], scoring_inputs[1])
#             self.verb_embeddings, self.code_embeddings = embedder.embed_batch(verb_embeddings[0], 
#                                                                               verb_embeddings[1], 
#                                                                               return_separators=self.return_separators)
            self.verb_embeddings, self.code_embeddings = embedder.embed_batch_v7(verb_embeddings[0], 
                                                                                 verb_embeddings[1], 
                                                                                 return_separators=self.return_separators)
            self.accumulated_loss = []
            _, output, _, _ = self.process_node(tree, code)
        except ProcessingException:
            return None  # todo: or return all zeros or something?

        inputs = embedder.get_feature_inputs_batch([" ".join(sample[0])], [" ".join(code)])
        inputs['weights'] = output[1]       
        pred = self.classifier(**inputs, output_hidden_states=True)
        return pred['logits']
