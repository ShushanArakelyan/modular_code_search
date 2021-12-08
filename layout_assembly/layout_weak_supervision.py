import codebert_embedder_v2 as embedder
from action.weak_supervision import weak_supervision_scores
from layout_assembly.layout_w_orig_verbs import LayoutNetWOrigVerbs
from layout_assembly.utils import ProcessingException

from nltk.stem import WordNetLemmatizer


class LayoutNet_weak_supervision(LayoutNetWOrigVerbs):
    def __init__(self, filter_func, scoring_module, action_module_facade, device, supervision_func=None,
                 is_sanity_check=False, use_cls_for_verb_emb=True, precomputed_scores_provided=False,
                 use_constant_for_weights=False):
        super().__init__(scoring_module, action_module_facade, device,
                         precomputed_scores_provided)
        self.filter = filter_func
        self.supervision_func = supervision_func
        self.is_sanity_check = is_sanity_check
        self.reverse_string2predicate = self.construct_reverse_string2predicate()
        self.lemmatizer = WordNetLemmatizer()
        self.use_cls_for_verb_emb = use_cls_for_verb_emb
        self.use_constant_for_weights = use_constant_for_weights

    def forward(self, ccg_parse, sample, orig_verbs=True):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        code = sample[1]
        if len(code) == 0:  # erroneous example
            return None
        scoring_inputs, verbs = self.precompute_inputs(tree, code, [[], [], []], [[], []], '')
        if len(verbs[0]) > 1:
            # too many actions
            return None
        if not self.filter(verbs[0][0].lower()):
            return None
        scoring_labels = self.scoring_module.forward_batch(scoring_inputs[0], scoring_inputs[1])
        try:
            self.verb_embeddings, self.code_embeddings = embedder.embed_in_list(verbs[0],
                                                                                verbs[1],
                                                                                return_cls_for_query=self.use_cls_for_verb_emb)
            if not self.is_sanity_check:
                _, output, _, _ = self.process_node(tree, code)
                output = output[1]
            else:
                output = weak_supervision_scores(embedder=embedder, code=code, verb=verbs[0][0],
                                                 attend_scores=scoring_labels,
                                                 matching_func=self.supervision_func)
        except ProcessingException:
            return None  # todo: or return all zeros or something?
        inputs, output = embedder.get_feature_inputs_classifier([" ".join(sample[0])], [" ".join(code)], output,
                                                                return_segment_ids=True)
        inputs['weights'] = output
        pred = self.classifier(**inputs, output_hidden_states=True)
        return pred['logits']
