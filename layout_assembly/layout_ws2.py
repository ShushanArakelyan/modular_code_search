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
    def __init__(self, scoring_module, action_module, device, code_in_output, weighted_cosine, mlp_prediction):
        print("in layout net: ", device)
        self.scoring_module = scoring_module
        self.action_module = action_module
        self.device = device
        self.finetune_codebert = True
        self.finetune_scoring = False
        self.code_in_output = code_in_output
        self.weighted_cosine = weighted_cosine
        self.mlp_prediction = mlp_prediction
        if self.weighted_cosine:
            self.weight = torch.autograd.Variable(torch.empty((768, 1), device=self.device), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.weight)
        if self.mlp_prediction:
            dim = embedder.dim
            self.distance_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim * 2, int(dim / 2)),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(int(dim / 2), 1),
                torch.nn.Sigmoid(),
            ).to(self.device)
        embedder.init_embedder(device)

    def forward(self, ccg_parse, sample):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        code = sample[1]
        # print("new example: ", ccg_parse)
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
        if self.weighted_cosine:
            return (outs[-1], self.weight)
        return outs[-1]

    def get_masking_idx(self):
        return 1

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
            # print("Num of inputs passed to action: ", len(action_module_wrapper.inputs))
            outputs = self.action_module.forward(action_module_wrapper.inputs, self.get_masking_idx(),
                                                 precomputed_embeddings)
            if self.code_in_output:
                outputs = outputs + (code_emb[action_it-1],)
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
        self.scoring_module.set_eval()
        embedder.model.eval()

    def set_train(self):
        self.action_module.set_train()
        self.scoring_module.set_train()
        embedder.model.train()

    def parameters(self):
        parameters = (self.action_module.parameters(),)
        if self.finetune_codebert:
            parameters = parameters + (embedder.model.parameters(),)
        if self.finetune_scoring:
            parameters = parameters + (self.scoring_module.parameters(),)
        if self.weighted_cosine:
            pass
            # parameters = parameters + (self.weight,)
        if self.mlp_prediction:
            parameters = parameters + (self.distance_mlp.parameters(),)
        return chain(*parameters)

    def named_parameters(self):
        parameters = (self.action_module.named_parameters(),)
        if self.finetune_codebert:
            parameters = parameters + (embedder.model.named_parameters(),)
        if self.finetune_scoring:
            parameters = parameters + (self.scoring_module.named_parameters(),)
        if self.weighted_cosine:
            parameters = parameters + (("weighted cosine weight", self.weight),)
        if self.mlp_prediction:
            parameters = parameters + (self.distance_mlp.named_parameters())
        return chain(*parameters)

    def load_from_checkpoint(self, checkpoint):
        self.action_module.load_from_checkpoint(checkpoint + '.action_module')
        try:
            self.scoring_module.load_from_checkpoint(checkpoint + '.scoring_module')
            print("LayoutNet: Successfully loaded scoring module checkpoint")
        except:
            print("LayoutNet: Could not load scoring module from checkpoint")
        models = torch.load(checkpoint, map_location=self.device)
        embedder.model.load_state_dict(models['codebert.model'])
        if 'weighted_cosine_weight' in models:
            self.weight = models["weighted_cosine_weight"]
        else:
            print("LayoutNet: Could not load weighted cosine weight from the checkpoint!")

    def save_to_checkpoint(self, checkpoint):
        self.action_module.save_to_checkpoint(checkpoint + '.action_module')
        if self.finetune_scoring:
            self.scoring_module.save_to_checkpoint(checkpoint + '.scoring_module')
        model_dict = {'codebert.model': embedder.model.state_dict()}
        if self.weighted_cosine:
            model_dict['weighted_cosine_weight'] = self.weight
        torch.save(model_dict, checkpoint)

    def state_dict(self):
        raise NotImplementedError()
