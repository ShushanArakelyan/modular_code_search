from itertools import chain

import torch

from layout_assembly.utils import ProcessingException, FC2


class ActionModule(object):
    def __init__(self, device, dim_size, dropout=0, max_inputs_allowed=3):
        self.device = device
        hidden_input_dims = [dim_size * 2, 512]
        hidden_output_dims = [512, 7]
        self.verb_embedder = FC2(hidden_input_dims, hidden_output_dims, dropout=self.dropout).to(self.device)
        self.cos = torch.nn.CosineSimilarity()
        self.modules = None
        self.max_inputs_allowed = max_inputs_allowed
        self.init_networks(dim_size, dropout)

    def init_networks(self, dim_size, dropout):
        for i in range(self.max_inputs_allowed):
            dim = dim_size + (i + 1) * 8
            self.modules[i] = torch.nn.Sequential(
                torch.nn.TransformerEncoderLayer(d_model=dim, nhead=8),
                torch.nn.Linear(dim, int(dim / 2)),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(int(dim / 2), 2)
            ).to(self.device)

    def forward(self, inputs, masking_indx, code, verb_embedding):
        updated_inputs = []
        num_inputs = len(inputs) - 1
        if num_inputs > 2:
            raise ProcessingException()
        module = self.modules[num_inputs]

        for indx, i in enumerate(inputs):
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
            repl_out = out.repeat(len(scores), 0)
            updated_i = torch.cat((repl_out, scores), dim=1)
            updated_inputs.append(updated_i)
        updated_inputs = torch.cat(updated_inputs)
        out_scores = module.forward(updated_inputs, code)

        return true_scores, out_scores

    def parameters(self):
        return chain([self.modules[i].parameters() for i in self.modules.keys()])

    def named_parameters(self):
        return chain([self.modules[i].named_parameters() for i in self.modules.keys()])

    def state_dict(self):
        state_dict = {f'{i}_input': self.modules[i].state_dict() for i in self.modules.keys()}
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
