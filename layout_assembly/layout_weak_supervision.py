import codebert_embedder_v2 as embedder
from layout_assembly.layout_codebert_classifier import LayoutNet_w_codebert_classifier
from layout_assembly.utils import ProcessingException


class LayoutNet_weak_supervision(LayoutNet_w_codebert_classifier):
    def __init__(self, filter_func, scoring_module, action_module_facade, device, use_cls_for_verb_emb=True,
                 precomputed_scores_provided=False, use_constant_for_weights=False):
        super().__init__(scoring_module, action_module_facade, device, use_cls_for_verb_emb,
                         precomputed_scores_provided, use_constant_for_weights)
        self.filter = filter_func

    def forward_with_sup_labels(self, ccg_parse, sample):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        code = sample[1]
        if len(code) == 0:  # erroneous example
            return None
        scoring_inputs, verbs = self.precompute_inputs(tree, code, [[], [], []], [[], []], '')
        if len(verbs[0]) > 1:
            # too many actions
            return None
        if not self.filter(verbs[0][0]):
            return None
        scoring_labels = self.scoring_module.forward_batch(scoring_inputs[0], scoring_inputs[1])
        try:
            self.verb_embeddings, self.code_embeddings = embedder.embed_in_list(verbs[0],
                                                                                verbs[1],
                                                                                return_cls_for_query=self.use_cls_for_verb_emb)
            _, output, _, _ = self.process_node(tree, code)
        except ProcessingException:
            return None  # todo: or return all zeros or something?
        inputs, output = embedder.get_feature_inputs_classifier([" ".join(sample[0])], [" ".join(code)], output[1],
                                                                return_segment_ids=True)
        inputs['weights'] = output
        pred = self.classifier(**inputs, output_hidden_states=True)
        return scoring_labels, pred['logits']
