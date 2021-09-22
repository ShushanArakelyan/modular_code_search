from itertools import chain

import torch
import numpy as np

import codebert_embedder_with_adapter as embedder
from layout_assembly.utils import ActionModuleWrapper, ProcessingException

from layout_assembly.layout import LayoutNode, LayoutNet

class LayoutNetWithAdapters(LayoutNet):
    def __init__(self, scoring_module, action_module_facade, device, precomputed_scores_provided=False, eval=False,
                 finetune_codebert=False):
        LayoutNet.__init__(scoring_module, action_module_facade, device, precomputed_scores_provided, eval, finetune_codebert)
        if not embedder.initialized:
            embedder.init_embedder(device)
        self.device = device
    
    def parameters(self):
        return chain(self.classifier.parameters(), self.action_module_facade.parameters(), embedder.param_generator.parameters())
    
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

    def forward(self, ccg_parse, sample):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        code = sample[1]
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
