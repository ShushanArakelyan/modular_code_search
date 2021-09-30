from itertools import chain

import torch
import numpy as np

import codebert_embedder as verb_embedder
import codebert_embedder_with_adapter as embedder
from layout_assembly.utils import ActionModuleWrapper, ProcessingException

from layout_assembly.layout import LayoutNode, LayoutNet

class LayoutNetWithAdapters(LayoutNet):
    def __init__(self, scoring_module, action_module_facade, device, precomputed_scores_provided=False, eval=False):
        super().__init__(scoring_module, action_module_facade, device, precomputed_scores_provided, eval)
        if not embedder.initialized:
            embedder.init_embedder(device)
        self.device = device
    
    def parameters(self):
        return chain(self.classifier.parameters(), self.action_module_facade.parameters(), embedder.param_generator.parameters())

    def named_parameters(self):
        return chain(self.classifier.named_parameters(), 
                     self.action_module_facade.named_parameters(), 
                     embedder.param_generator.named_parameters())
    
    def load_from_checkpoint(self, checkpoint):
        self.action_module_facade.load_from_checkpoint(checkpoint + '.action_module')

        models = torch.load(checkpoint, map_location=self.device)
        self.classifier.load_state_dict(models['classifier'])
        self.classifier = self.classifier.to(self.device)
        embedder.param_generator.load_state_dict(models['embedder_param_generator'])
        embedder.param_generator = embedder.param_generator.to(self.device)

    def save_to_checkpoint(self, checkpoint):
        self.action_module_facade.save_to_checkpoint(checkpoint + '.action_module')
        torch.save({'classifier': self.classifier.state_dict(), 
                   'embedder_param_generator': embedder.param_generator.state_dict()}, checkpoint)

    def state_dict(self):
        return {'action_module': self.action_module_facade.state_dict(), 
                'classifier': self.classifier.state_dict(), 
                'embedder_param_generator': embedder.param_generator.state_dict()}

    def use_adapter(self, use_adapter):
        embedder.model.use_adapter(use_adapter)
    
    def forward(self, ccg_parse, sample, use_adapter=True):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        code = sample[1]
        if not self.precomputed_scores_provided: 
            scoring_inputs, verb_embeddings = self.precompute_inputs(tree, code, [[], [], []], [[], []], '')
            self.scoring_outputs = self.scoring_module.forward_batch(scoring_inputs[0], scoring_inputs[1])
            self.verb_embeddings, _ = verb_embedder.embed_batch(verb_embeddings[0], verb_embeddings[1])
        self.code_embeddings = []
        for ve in self.verb_embeddings:
            self.code_embeddings.append(embedder.embed(ve, ' '.join(code)))
        self.code_embeddings = torch.cat(self.code_embeddings, dim=0)
        try:
            _, output, _, _ = self.process_node(tree, code)
        except ProcessingException:
            return None  # todo: or return all zeros or something?
        pred = self.classifier.forward(output[0])
        return pred
