from itertools import chain

import torch

import codebert_embedder_v2 as embedder
from layout_assembly.layout import LayoutNet
from layout_assembly.utils import ProcessingException


# this copy here is necessary because otherwise we use an incorrect instance of codebert embedder
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

    def set_eval(self):
        self.module.set_eval()

    def set_train(self):
        self.module.set_train()


class LayoutNet_w_codebert_classifier(LayoutNet):
    def __init__(self, scoring_module, action_module_facade, device, return_separators=False,
                 precomputed_scores_provided=False, embed_in_list=False):
        print(device)
        self.scoring_module = scoring_module
        self.action_module_facade = action_module_facade
        self.device = device
        self.precomputed_scores_provided = precomputed_scores_provided
        self.return_separators = return_separators
        self.classifier = embedder.classifier
        self.embed_in_list = embed_in_list
        self.scoring_outputs = None
        self.accumulated_loss = None

    def parameters(self):
        return chain(self.classifier.parameters(), self.action_module_facade.parameters())

    def load_from_checkpoint(self, checkpoint):
        self.action_module_facade.load_from_checkpoint(checkpoint + '.action_module')

        models = torch.load(checkpoint, map_location=self.device)
        self.classifier.load_state_dict(models['classifier'])
        self.classifier = self.classifier.to(self.device)
        if 'codebert.model' in models:
            print("Loading CodeBERT weights from the checkpoint")
            embedder.model.load_state_dict(models['codebert.model'])

    def save_to_checkpoint(self, checkpoint):
        self.action_module_facade.save_to_checkpoint(checkpoint + '.action_module')
        model_dict = {'classifier': self.classifier.state_dict()}
        model_dict['codebert.model'] = embedder.model.state_dict()
        torch.save(model_dict, checkpoint)

    def state_dict(self):
        return {'action_module': self.action_module_facade.state_dict(), 'classifier': self.classifier.state_dict()}

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
            if not self.embed_in_list:
                self.verb_embeddings, self.code_embeddings = embedder.embed_batch(verb_embeddings[0],
                                                                                  verb_embeddings[1],
                                                                                  return_separators=self.return_separators)
            else:
                self.verb_embeddings, self.code_embeddings = embedder.embed_in_list(verb_embeddings[0],
                                                                                    verb_embeddings[1])
            self.accumulated_loss = []
            _, output, _, _ = self.process_node(tree, code)
        except ProcessingException:
            return None  # todo: or return all zeros or something?
        print("Scores: ", output[1])
        inputs, scores = embedder.get_feature_inputs_classifier([" ".join(sample[0])], [" ".join(code)], scores)
        inputs['weights'] = output[1]
        pred = self.classifier(**inputs, output_hidden_states=True)
        return pred['logits']

    def process_node(self, node, code, scoring_it=0, action_it=0, parent_module=None):
        if node.node_type == 'action':
            action_module = ActionModuleWrapper(self.action_module_facade)
            action_module.param = node.node_value
            for child in node.children:
                action_module, _, scoring_it, action_it = self.process_node(child, code, scoring_it, action_it,
                                                                            action_module)
            precomputed_embeddings = (self.verb_embeddings[action_it], self.code_embeddings[action_it])
            if precomputed_embeddings[0].shape[0] == 0 or precomputed_embeddings[1].shape[0] == 0:
                raise ProcessingException()
            action_it += 1
            output = action_module.forward(code, precomputed_embeddings)
            self.accumulated_loss.append(output[-1])
            output = (output[0], output[1])  # remove last element
            if parent_module:
                parent_module.add_input(output)
            return parent_module, output, scoring_it, action_it
        elif node.node_type == 'scoring':
            output = self.scoring_outputs[scoring_it]
            if output.shape[0] == 0:
                raise ProcessingException()
            scoring_it += 1
            parent_module.add_input(output)
            return parent_module, output, scoring_it, action_it
        elif node.node_type == 'preposition':
            parent_module.add_preposition(node.node_value)
            for child in node.children:
                self.process_node(child, code, scoring_it, action_it, parent_module)
            return parent_module, None, scoring_it, action_it
