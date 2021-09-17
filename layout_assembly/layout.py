from itertools import chain

import torch

import codebert_embedder as embedder
from layout_assembly.utils import ActionModuleWrapper
from layout_assembly.utils import ProcessingException


class LayoutNode:
    def __init__(self):
        self.node_type = None
        self.node_value = None
        self.children = []
        self.parent = None


class LayoutNet:
    def __init__(self, scoring_module, action_module_facade, device, eval=False):
        self.scoring_module = scoring_module
        self.action_module_facade = action_module_facade
        self.device = device
        dim = embedder.dim
        half_dim = int(dim / 2)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(dim, half_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(half_dim, 1)).to(device)
        self.scoring_outputs = None
        self.eval = eval
        if self.eval:
            self.classifier.eval()

    def parameters(self):
        return chain(self.classifier.parameters(), self.action_module_facade.parameters())

    def load_from_checkpoint(self, checkpoint):
        self.action_module_facade.load_from_checkpoint(checkpoint + '.action_module')

        models = torch.load(checkpoint, map_location=self.device)
        self.classifier.load_state_dict(models['classifier'])
        self.classifier = self.classifier.to(self.device)

    def save_to_checkpoint(self, checkpoint):
        self.action_module_facade.save_to_checkpoint(checkpoint + '.action_module')
        torch.save({'classifier': self.classifier.state_dict()}, checkpoint)

    def state_dict(self):
        return {'action_module': self.action_module_facade.state_dict(), 'classifier': self.classifier.state_dict()}

    def forward(self, ccg_parse, sample):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        scoring_inputs, verb_embeddings = self.precompute_inputs(tree, sample[1], [[], [], []], [[], []], '')
        self.scoring_outputs = self.scoring_module.forward_batch(scoring_inputs[0], scoring_inputs[1])
        self.verb_embeddings, self.code_embeddings = embedder.embed_batch(verb_embeddings[0], verb_embeddings[1])
        try:
            _, output, _ = self.process_node(tree)
        except ProcessingException:
            return None  # todo: or return all zeros or something?
        pred = self.classifier.forward(output[0])
        return pred

    def process_node(self, node, scoring_it=0, action_it=0, parent_module=None):
        if node.node_type == 'action':
            action_module = ActionModuleWrapper(self.action_module_facade)
            action_module.param = node.node_value
            for child in node.children:
                action_module, _, scoring_it, action_it = self.process_node(child, scoring_it, action_it, action_module)
            output = (self.verb_embeddings[action_it], self.code_embeddings[action_it])
            action_it += 1
            action_module.forward(code, output)
            if  :
                parent_module.add_input(output)
            return parent_module, output, scoring_it, action_it
        elif node.node_type == 'scoring':
            output = self.scoring_outputs[scoring_it]
            scoring_it += 1
            parent_module.add_input(output)
            return parent_module, output, scoring_it, action_it
        elif node.node_type == 'preposition':
            parent_module.add_preposition(node.node_value)
            for child in node.children:
                self.process_node(child, scoring_it, action_it, parent_module)
            return parent_module, None, scoring_it, action_it

    def precompute_inputs (self, node, code, scoring_inputs, verb_embeddings, param=None):
        if node.node_type == 'action':
            for child in node.children:
                scoring_inputs, verb_embeddings = self.precompute_inputs(child, code, scoring_inputs, verb_embeddings, node.node_value)
            verb_embeddings[0].append(node.node_value)
            verb_embeddings[1].append(code)
        elif node.node_type == 'scoring':
            scoring_inputs[0].append(node.node_value)
            scoring_inputs[1].append(code)
            scoring_inputs[2].append(param)
        elif node.node_type == 'preposition':
            for child in node.children:
                scoring_inputs, verb_embeddings = self.precompute_inputs(child, code, scoring_inputs, verb_embeddings, param)
        return scoring_inputs, verb_embeddings

    @staticmethod
    def remove_concats(tree):
        stack = [tree]
        while stack:
            node = stack.pop()
            new_children = []
            new_node = None
            for child in node.children:
                if child.node_type == 'scoring':
                    if new_node is None:
                        new_node = LayoutNode()
                        new_node.node_type = 'scoring'
                        new_node.parent = node.parent
                        new_node.node_value = child.node_value
                    else:
                        new_node.node_value += ' ' + child.node_value
                else:
                    if new_node:
                        new_children.append(new_node)
                        new_node = None
                    new_children.append(child)
            if new_node:
                new_children.append(new_node)
            node.children = new_children
            stack.extend(new_children)
        return tree

    @staticmethod
    def construct_layout(ccg_parse):
        parts = ccg_parse.split(' ')[::-1]
        tree = LayoutNode()
        node = tree
        stack = []
        parent = None
        while parts:
            current_part = parts.pop()
            if len(current_part) == 0:
                continue
            if current_part == '@Action':
                node.node_type = 'action'
                parts.pop()  # opening bracket
                node.node_value = parts.pop()
                node.parent = parent
                stack.append(node)
                if parent:
                    parent.children.append(node)
                parent = node
                node = LayoutNode()
            elif current_part == '@Concat' or current_part == '@Num':
                node.node_type = 'scoring'
                parts.pop()  # opening bracket
                node.node_value = parts.pop()
                node.parent = parent
                stack.append(node)
                parent.children.append(node)

                node = LayoutNode()
            elif current_part.startswith('@'):
                node.node_type = 'preposition'
                node.node_value = current_part[1:].lower()
                parts.pop()  # opening bracket
                node.parent = parent
                stack.append(node)
                parent.children.append(node)

                parent = node
                node = LayoutNode()
            elif current_part.startswith(')'):
                current_part = list(current_part)
                while current_part:
                    stack.pop()
                    current_part.pop()
                if stack:
                    parent = stack[-1]
            else:
                node.node_type = 'scoring'
                node.node_value = current_part

                node.parent = parent
                parent.children.append(node)

                node = LayoutNode()
        return tree
