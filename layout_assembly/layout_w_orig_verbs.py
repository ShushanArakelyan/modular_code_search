from itertools import chain

import torch

import codebert_embedder as embedder
from layout_assembly.utils import ActionModuleWrapper, ProcessingException, init_weights
from layout_assembly.layout import LayoutNet

import numpy as np
import stanza
import sys

sys.path.append('/home/shushan/')
from ccg_parser import parser_dict
from ccg_parser.parser import get_wordnet_pos
from nltk.stem import WordNetLemmatizer


POS_TAGGER = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
class LayoutNetWOrigVerbs(LayoutNet):
    def __init__(self, scoring_module, action_module_facade, device, precomputed_scores_provided=False, eval=False, finetune_codebert=True):
        print(device)
        self.scoring_module = scoring_module
        self.action_module_facade = action_module_facade
        self.device = device
        self.finetune_codebert = finetune_codebert
        self.precomputed_scores_provided = precomputed_scores_provided
        self.reverse_string2predicate = self.construct_reverse_string2predicate()
#         print("Constructed reverse string 2 predicate: ", self.reverse_string2predicate)
        self.lemmatizer = WordNetLemmatizer()
        dim = embedder.dim
        half_dim = int(dim / 2)

        self.classifier = torch.nn.Sequential(torch.nn.Linear(dim, half_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(half_dim, 1)).to(self.device)
        self.classifier.apply(init_weights)
        self.scoring_outputs = None
        self.accumulated_loss = None
        self.eval = eval
        if self.eval:
            self.classifier.eval()

    def forward(self, ccg_parse, sample):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        code = sample[1]
        if len(code) == 0:  # erroneous example
            return None
        try:
            scoring_inputs, verb_embeddings = self.precompute_inputs(tree, code, [[], [], []], [[], []], '')
            if np.any(np.unique(verb_embeddings[0], return_counts=True)[1] > 1):
#                 print("np.unique(verb_embeddings[0], return_counts=True)[1]: ", np.unique(verb_embeddings[0], return_counts=True)[1])
#                 print(verb_embeddings[0])
                return None
            verb_embeddings[0] = [self.get_orig_verb(v, sample[0]) for v in verb_embeddings[0]]
#             print("verbs in forward: ", verb_embeddings[0])
            if not self.precomputed_scores_provided:
                self.scoring_outputs = self.scoring_module.forward_batch(scoring_inputs[0], scoring_inputs[1])
            self.verb_embeddings, self.code_embeddings = embedder.embed_batch(verb_embeddings[0], verb_embeddings[1])
            self.accumulated_loss = []
            _, output, _, _ = self.process_node(tree, code, sample[0])
        except ProcessingException:
            return None  # todo: or return all zeros or something?
        pred = self.classifier.forward(output[0])
        return pred

    def process_node(self, node, code, orig_sentence, scoring_it=0, action_it=0, parent_module=None):
        if node.node_type == 'action':
            action_module = ActionModuleWrapper(self.action_module_facade)
            action_module.param = self.get_orig_verb(node.node_value, orig_sentence)
            for child in node.children:
                action_module, _, scoring_it, action_it = self.process_node(child, code, orig_sentence, scoring_it, action_it,
                                                                            action_module)
            precomputed_embeddings = (self.verb_embeddings[action_it], self.code_embeddings[action_it])
            if precomputed_embeddings[0].shape[0] == 0 or precomputed_embeddings[1].shape[0] == 0:
                raise ProcessingException()
            action_it += 1
            output = action_module.forward(code, precomputed_embeddings)
            self.accumulated_loss.append(output[-1])
            output = (output[0], output[1]) # remove last element
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
                self.process_node(child, code, orig_sentence, scoring_it, action_it, parent_module)
            return parent_module, None, scoring_it, action_it

    def precompute_inputs(self, node, code, scoring_inputs, verb_embeddings, param=None):
        if node.node_type == 'action':
            for child in node.children:
                scoring_inputs, verb_embeddings = self.precompute_inputs(child, code, scoring_inputs, verb_embeddings,
                                                                         node.node_value)
            verb_embeddings[0].append(node.node_value)
            verb_embeddings[1].append(code)
        elif node.node_type == 'scoring':
            scoring_inputs[0].append(node.node_value)
            scoring_inputs[1].append(code)
            scoring_inputs[2].append(param)
        elif node.node_type == 'preposition':
            for child in node.children:
                scoring_inputs, verb_embeddings = self.precompute_inputs(child, code, scoring_inputs, verb_embeddings,
                                                                         param)
        return scoring_inputs, verb_embeddings

    def get_orig_verb(self, verb, orig_sentence):
        if verb not in self.reverse_string2predicate:
            return verb
        orig_tokens = POS_TAGGER([orig_sentence]).sentences[0].words
        all_verbs = self.reverse_string2predicate[verb]
#         print("all verbs for verb: ", verb, ' ', all_verbs)                                                  
        for token in orig_tokens:
#             print(token.text.lower())
            if token.text.lower() in all_verbs:
#                 print(f"replacing verb {verb} with {token.text.lower()}")
                return token.text.lower()
            else:
                lemma_form = self.lemmatizer.lemmatize(token.text.lower(), get_wordnet_pos(token.xpos)) 
                if lemma_form in all_verbs:
#                     print(f"replacing verb {verb} with {lemma_form}")
                    return lemma_form
        return 'load' # in some sentences we cannot identify the verb, and replace it with a 'load'
    
    @staticmethod
    def construct_reverse_string2predicate():
        predicate2strings = {}
        for string, predicates in parser_dict.STRING2PREDICATE.items():
            for predicate in predicates:
                if predicate.startswith('$'):
                    predicate = predicate[1:]
                if predicate in predicate2strings:
                    predicate2strings[predicate].append(string.lower())
                else:
                    predicate2strings[predicate] = [string.lower()]
        return predicate2strings