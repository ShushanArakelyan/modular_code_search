from layout_assembly.utils import ActionModuleWrapper
from layout_assembly.utils import ProcessingException

import codebert_embedder as embedder
from layout_assembly.layout import LayoutNet
import torch


class LayoutNet_v2(LayoutNet):
    def forward(self, ccg_parse, sample):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        try:
            with torch.no_grad:
                code_embeddings = embedder.embed(' ', sample[1])
                padding_size = embedder.max_seq_length - len(code_embeddings)
                code_embeddings = torch.nn.functional.pad(code_embeddings, (0, 0, 0, padding_size), 'constant', 0)
            _, output = self.process_node(tree, sample, code_embeddings)
        except ProcessingException:
            return None  # todo: or return all zeros or something?
        pred = self.classifier.forward(output[0])
        return pred

    def process_node(self, node, sample, code_embeddings, parent_module=None):
        if node.node_type == 'action':
            action_module = ActionModuleWrapper(self.action_module_facade)
            action_module.param = node.node_value
            for child in node.children:
                action_module, _ = self.process_node(child, sample, code_embeddings, action_module)
            output = action_module.forward(code_embeddings)
            if parent_module:
                parent_module.add_input(output)
            return parent_module, output
        elif node.node_type == 'scoring':
            output = self.scoring_module.forward(node.node_value, sample)
            parent_module.add_input(output)
            return parent_module, output
        elif node.node_type == 'preposition':
            parent_module.add_preposition(node.node_value)
            for child in node.children:
                self.process_node(child, sample, code_embeddings, parent_module)
            return parent_module, None