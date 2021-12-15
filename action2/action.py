from itertools import chain

import torch

from layout_assembly.utils import ProcessingException, FC2


class ActionModule(object):
    def __init__(self, device, dim_size, dropout=0, max_inputs_allowed=3):
        self.device = device
        hidden_input_dims = [dim_size * 2, 512]
        hidden_output_dims = [512, 7]
        self.verb_embedder = FC2(hidden_input_dims, hidden_output_dims, dropout=dropout).to(self.device)
        self.cos = torch.nn.CosineSimilarity()
        self.modules = None
        self.max_inputs_allowed = max_inputs_allowed
        self.init_networks(dim_size, dropout)

    def init_networks(self, dim_size, dropout):
        self.modules = {}
        for i in range(self.max_inputs_allowed):
            dim = dim_size + (i + 1) * 8
            self.modules[i] = torch.nn.Sequential(
                torch.nn.TransformerEncoderLayer(d_model=dim, nhead=8),
                torch.nn.Linear(dim, int(dim / 2)),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(int(dim / 2), 1)
            ).to(self.device)

    def forward(self, inputs, masking_indx, precomputed_embeddings):
        updated_inputs = []
        num_inputs = len(inputs) - 1
        if num_inputs > 2 or precomputed_embeddings is None:
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
                scores = torch.zeros_like(scores).to(self.device)
            fwd_input = torch.cat((verb_embedding, prep_embedding), dim=1)
            out = self.verb_embedder(fwd_input)
            repl_out = out.repeat(len(scores), 1)
            updated_i = torch.cat((repl_out, scores), dim=1)
            updated_inputs.append(updated_i)
        module = self.modules[num_inputs]
        N = min([u.shape[0] for u in updated_inputs])
        N = min(N, code_embedding.shape[0])
        # scores might be of different sizes depending on the query they were embedded with.
        updated_inputs = [u[:N, :] for u in updated_inputs]
        updated_inputs = torch.cat(updated_inputs, dim=1)
        code_embedding = code_embedding[:N, :]
        final_fwd_input = torch.cat((updated_inputs, code_embedding), dim=1).unsqueeze(dim=1)
        out_scores = module.forward(final_fwd_input).squeeze(dim=1)
        N = min(len(true_scores), len(out_scores))
        true_scores = true_scores[:N, :]
        out_scores = out_scores[:N, :]
        return true_scores, out_scores

    def parameters(self):
        return chain.from_iterable([self.modules[i].parameters() for i in self.modules.keys()] +
                                   [self.verb_embedder.parameters()])

    def named_parameters(self):
        return chain.from_iterable([self.modules[i].named_parameters() for i in self.modules.keys()] +
                                   [self.verb_embedder.named_parameters()])

    def state_dict(self):
        state_dict = {f'{i}_input': self.modules[i].state_dict() for i in self.modules.keys()}
        state_dict['verb_embedder'] = self.verb_embedder.state_dict()
        return state_dict

    def save_to_checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def load_from_checkpoint(self, checkpoint):
        raise NotImplementedError()
        # models = torch.load(checkpoint, map_location=self.device)
        #
        # self.one_input_module.load_state_dict(models['one_input'])
        # self.two_inputs_module.load_state_dict(models['two_inputs'])

    def set_eval(self):
        for i, m in self.modules.iteritems():
            m.eval()

    def set_train(self):
        for i, m in self.modules.iteritems():
            m.train()
