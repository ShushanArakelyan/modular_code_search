import torch

from action2.action import ActionModule
from layout_assembly.utils import ProcessingException

class DummyActionModule(ActionModule):

    def forward(self, inputs, masking_indx, precomputed_embeddings):
        dummy_outputs = []
        num_inputs = len(inputs) - 1
        if num_inputs > self.max_inputs_allowed - 1 or precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embedding = precomputed_embeddings
        for indx, i in enumerate(inputs):
            if len(i) < 2:
                num_inputs -= 1
                # we are skipping some arguments, e.g. action-s, so it is possible
                # to have a preposition without its corresponding scores
                continue
            prep_embedding, scores = i
            if isinstance(scores, tuple):
                _, scores = scores
            if len(scores.shape) == 1:
                scores = scores.unsqueeze(dim=1)
            if indx == masking_indx:
                # mask this index
                true_scores = scores
                dummy_out = torch.zeros_like(scores).to(self.device)
            else:
                dummy_out = torch.ones_like(scores).to(self.device)
            dummy_outputs.append(dummy_out)
        if num_inputs < 0:
            raise ProcessingException()
        N = min([o.shape[0] for o in dummy_outputs])
        N = min(N, code_embedding.shape[0])
        truncated_true_scores = true_scores[:N, :]
        dummy_out_scores = torch.ones(N, 1).to(self.device)
        return truncated_true_scores, dummy_out_scores