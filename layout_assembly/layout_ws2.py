from itertools import chain

import numpy as np
import torch

import codebert_embedder_v2 as embedder
from layout_assembly.layout import LayoutNet
from layout_assembly.utils import ProcessingException


class ActionModuleWrapper(object):
    def __init__(self, action_module_facade, device):
        self.empty_emb = None
        self.prep_emb_cache = {}
        self.param = None
        self.inputs = []
        self.input_count = 0
        self.device = device
        self.module = action_module_facade

    def forward(self, code, verb_embedding):
        return self.module.forward(self.param, self.inputs, code, verb_embedding)

    def add_input(self, input):
        if len(self.inputs) > 0 and len(self.inputs[-1]) == 1:
            self.inputs[-1].append(input)
        else:
            with torch.no_grad():
                if self.empty_emb is None:
                    self.empty_emb = torch.zeros(1, embedder.dim, requires_grad=False).to(self.device)
            self.inputs.append([self.empty_emb, input])

    def add_preposition(self, prep):
        if prep not in self.prep_emb_cache:
            self.prep_emb_cache[prep] = embedder.embed([prep], [' '])[1]
        self.inputs.append([self.prep_emb_cache[prep]])

    def set_eval(self):
        self.module.set_eval()

    def set_train(self):
        self.module.set_train()


class LayoutNetWS2(LayoutNet):
    def __init__(self, scoring_module, action_module, device):
        print("in layout net: ", device)
        self.scoring_module = scoring_module
        self.action_module = action_module
        self.device = device
        self.finetune_codebert = True
        self.finetune_scoring = False
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
        if self.finetune_scoring:
            scoring_forward_method = self.scoring_module.forward_batch
        else:
            scoring_forward_method = self.scoring_module.forward_batch_no_grad

        scoring_outputs = scoring_forward_method(scoring_inputs[0], scoring_inputs[1])
        verb_embeddings, code_embeddings = embedder.embed_in_list(verb_embeddings[0], verb_embeddings[1])
        outs = self.process_node(tree, scoring_outputs, verb_embeddings, code_embeddings, output_list=[],
                                 scoring_it=0, action_it=0)
        return outs[-1]

    def get_masking_idx(self):
        return 0

    def process_node(self, node, scoring_emb, verb_emb, code_emb, output_list, scoring_it=0, action_it=0,
                     parent_module=None):
        if node.node_type == 'action':
            action_module_wrapper = ActionModuleWrapper(self.action_module, self.device)
            action_module_wrapper.param = node.node_value
            for child in node.children:
                action_module_wrapper, scoring_it, action_it, output_list = self.process_node(node=child,
                                                                                              scoring_emb=scoring_emb,
                                                                                              verb_emb=verb_emb,
                                                                                              code_emb=code_emb,
                                                                                              scoring_it=scoring_it,
                                                                                              action_it=action_it,
                                                                                              output_list=output_list,
                                                                                              parent_module=action_module_wrapper)
            precomputed_embeddings = (verb_emb[action_it], code_emb[action_it])
            if precomputed_embeddings[0].shape[0] == 0 or precomputed_embeddings[1].shape[0] == 0:
                raise ProcessingException()
            action_it += 1
            outputs = self.action_module.forward(action_module_wrapper.inputs, self.get_masking_idx(),
                                                 precomputed_embeddings)
            output_list.append(outputs)
            return parent_module, scoring_it, action_it, output_list
        elif node.node_type == 'scoring':
            output = scoring_emb[scoring_it]
            if output.shape[0] == 0:
                raise ProcessingException()
            scoring_it += 1
            parent_module.add_input(output)
            return parent_module, scoring_it, action_it, output_list
        elif node.node_type == 'preposition':
            parent_module.add_preposition(node.node_value)
            for child in node.children:
                self.process_node(node=child,
                                  scoring_emb=scoring_emb,
                                  verb_emb=verb_emb,
                                  code_emb=code_emb,
                                  scoring_it=scoring_it,
                                  action_it=action_it,
                                  output_list=output_list,
                                  parent_module=parent_module)
            return parent_module, scoring_it, action_it, output_list

    def set_eval(self):
        self.action_module.set_eval()

    def set_train(self):
        self.action_module.set_train()

    def parameters(self):
        parameters = (self.action_module.parameters(),)
        if self.finetune_codebert:
            parameters = parameters + (embedder.model.parameters(),)
        if self.finetune_scoring:
            parameters = parameters + (self.scoring_module.parameters(),)
        return chain(*parameters)

    def named_parameters(self):
        parameters = (self.action_module.named_parameters(),)
        if self.finetune_codebert:
            parameters = parameters + (embedder.model.named_parameters(),)
        if self.finetune_scoring:
            parameters = parameters + (self.scoring_module.named_parameters(),)
        return chain(*parameters)

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
