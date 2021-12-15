from itertools import chain

import numpy as np
import torch

import codebert_embedder as embedder
from layout_assembly.layout import LayoutNet
from layout_assembly.utils import ActionModuleWrapper, ProcessingException


class LayoutNetWS2(LayoutNet):
    def __init__(self, scoring_module, action_module, device):
        print(device)
        self.scoring_module = scoring_module
        self.action_module = action_module
        self.device = device
        self.scoring_outputs = None
        self.finetune_codebert = True
        embedder.init_embedder(device)

    def forward(self, ccg_parse, sample):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        code = sample[1]
        if len(code) == 0:  # erroneous example
            raise ProcessingException()
        scoring_inputs, verb_embeddings = self.precompute_inputs(tree, code, [[], [], []], [[], []], '')
        if np.any(np.unique(verb_embeddings[0], return_counts=True)[1] > 1):
            raise ProcessingException()
        self.scoring_outputs = self.scoring_module.forward_batch(scoring_inputs[0], scoring_inputs[1])
        self.verb_embeddings, self.code_embeddings = embedder.embed_batch(verb_embeddings[0], verb_embeddings[1])
        outs = self.process_node(tree, code, sample[0])
        return outs[-1]

    def get_masking_idx(self):
        return 1

    def process_node(self, node, code, orig_sentence, scoring_it=0, action_it=0, output_list=[], parent_module=None):
        if node.node_type == 'action':
            action_module_wrapper = ActionModuleWrapper(self.action_module)
            action_module_wrapper.param = self.get_orig_verb(node.node_value, orig_sentence)
            for child in node.children:
                action_module_wrapper, scoring_it, action_it, output_list = self.process_node(child, code,
                                                                                              orig_sentence,
                                                                                              scoring_it, action_it,
                                                                                              output_list,
                                                                                              action_module_wrapper)
            precomputed_embeddings = (self.verb_embeddings[action_it], self.code_embeddings[action_it])
            if precomputed_embeddings[0].shape[0] == 0 or precomputed_embeddings[1].shape[0] == 0:
                raise ProcessingException()
            action_it += 1
            outputs = self.action_module.forward(action_module_wrapper.inputs, code, self.get_masking_idx(),
                                                 precomputed_embeddings)
            output_list.append(outputs)
            return parent_module, scoring_it, action_it, output_list
        elif node.node_type == 'scoring':
            output = self.scoring_outputs[scoring_it]
            if output.shape[0] == 0:
                raise ProcessingException()
            scoring_it += 1
            parent_module.add_input(output)
            return parent_module, scoring_it, action_it, output_list
        elif node.node_type == 'preposition':
            parent_module.add_preposition(node.node_value)
            for child in node.children:
                self.process_node(child, code, orig_sentence, scoring_it, action_it, output_list, parent_module)
            return parent_module, scoring_it, action_it, output_list

    def set_eval(self):
        self.action_module.set_eval()

    def set_train(self):
        self.action_module.set_train()

    def parameters(self):
        if self.finetune_codebert:
            return chain(self.action_module.parameters(), embedder.model.parameters())
        else:
            return chain(self.action_module.parameters())

    def load_from_checkpoint(self, checkpoint):
        # self.action_module_facade.load_from_checkpoint(checkpoint + '.action_module')
        #
        # models = torch.load(checkpoint, map_location=self.device)
        # self.classifier.load_state_dict(models['classifier'])
        # self.classifier = self.classifier.to(self.device)
        # if 'codebert.model' in models:
        #     print("Loading CodeBERT weights from the checkpoint")
        #     embedder.model.load_state_dict(models['codebert.model'])
        pass

    def save_to_checkpoint(self, checkpoint):
        self.action_module.save_to_checkpoint(checkpoint + '.action_module')
        model_dict = {'codebert.model': embedder.model.state_dict()}
        torch.save(model_dict, checkpoint)

    def state_dict(self):
        return {'action_module': self.action_module.state_dict()}
