from itertools import chain

import torch

from layout_assembly.utils import ProcessingException, FC2
import numpy as np


class ActionModule(object):
    def __init__(self, device, dim_size, dropout=0, max_inputs_allowed=3):
        self.device = device
        hidden_input_dims = [dim_size * 2, 512]
        hidden_output_dims = [512, 7]
        self.verb_embedder = FC2(hidden_input_dims, hidden_output_dims, dropout=dropout).to(self.device)
        self.modules = None
        self.max_inputs_allowed = max_inputs_allowed
        self.init_networks(dim_size, dropout)
        self.dim_size = dim_size
        self.dropout = dropout

    def init_networks(self, dim_size, dropout):
        self.modules = {}
        for i in range(self.max_inputs_allowed):
            dim = dim_size + (i + 1) * 8
            self.modules[i] = torch.nn.Sequential(
                torch.nn.TransformerEncoderLayer(d_model=dim, nhead=8),
                torch.nn.Linear(dim, int(dim / 2)),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(int(dim / 2), 1),
                torch.nn.Sigmoid(),
            ).to(self.device)

    def forward(self, inputs, masking_indx, precomputed_embeddings):
        updated_inputs = []
        num_unmasked_inputs = len(inputs) - 1
        # print("Num unmasked inputs: ", num_unmasked_inputs)
        if num_unmasked_inputs > self.max_inputs_allowed - 1 or precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embedding = precomputed_embeddings

        # check if there are any action arguments, which we don't include as argument
        for indx, i in enumerate(inputs):
            if len(i) < 2:
                num_unmasked_inputs -= 1
                continue
        masking_indx = min(masking_indx, num_unmasked_inputs)
        # print("Masking indx: ", masking_indx)
        masked = False
        for indx, i in enumerate(inputs):
            if len(i) < 2: # check if there are any action arguments, which we don't include as argument
                continue
            prep_embedding, scores = i
            if isinstance(scores, tuple):
                _, scores = scores
            if len(scores.shape) == 1:
                scores = scores.unsqueeze(dim=1)
            if indx == masking_indx:
                masked = True
                # print("Masking current idx: ", indx)
                # mask this index
                true_scores = scores
                if not np.any(true_scores.detach().cpu().numpy()):
                    print("all output scores are zeros")
                scores = torch.zeros_like(scores).to(self.device)
            fwd_input = torch.cat((verb_embedding, prep_embedding), dim=1)
            out = self.verb_embedder(fwd_input)
            repl_out = out.repeat(len(scores), 1)
            updated_i = torch.cat((repl_out, scores), dim=1)
            updated_inputs.append(updated_i)
        if not masked: # stupid workaround
            raise ProcessingException()
        if num_unmasked_inputs < 0:
            raise ProcessingException()
        module = self.modules[num_unmasked_inputs]
        N = min([u.shape[0] for u in updated_inputs])
        N = min(N, code_embedding.shape[0])
        # scores might be of different sizes depending on the query they were embedded with.
        updated_inputs = [u[:N, :] for u in updated_inputs]
        updated_inputs = torch.cat(updated_inputs, dim=1)
        code_embedding = code_embedding[:N, :]
        final_fwd_input = torch.cat((updated_inputs, code_embedding), dim=1).unsqueeze(dim=1)
        out_scores = module.forward(final_fwd_input).squeeze(dim=1)
        N = min(len(true_scores), len(out_scores))
        truncated_true_scores = true_scores[:N, :]
        truncated_out_scores = out_scores[:N, :]
        return truncated_true_scores, truncated_out_scores

    def parameters(self):
        return chain([param for i, m in self.modules.items() for param in m.parameters()],
                     self.verb_embedder.parameters())

    def named_parameters(self):
        return chain([(f"{i}_module.{name}", param) for i, m in self.modules.items() for name, param in m.named_parameters()],
                     [(f"verb_embedder.{name}", param) for name, param in self.verb_embedder.named_parameters()])

    def state_dict(self):
        state_dict = {f'{i}_input': self.modules[i].state_dict() for i in self.modules.keys()}
        state_dict['verb_embedder'] = self.verb_embedder.state_dict()
        return state_dict

    def save_to_checkpoint(self, checkpoint):
        save_dict = {i: m.state_dict() for i, m in self.modules.items()}
        save_dict['verb_embedder'] = self.verb_embedder.state_dict()
        save_dict['dim_size'] = self.dim_size
        save_dict['dropout'] = self.dropout
        torch.save(save_dict, checkpoint)

    def load_from_checkpoint(self, checkpoint):
        save_dict = torch.load(checkpoint, map_location=self.device)
        self.dropout = save_dict['dropout']
        self.dim_size = save_dict['dim_size']
        self.max_inputs_allowed = len(save_dict) - 3
        print("Loading from checkpoint, max inputs allowed ", self.max_inputs_allowed)
        self.init_networks(self.dim_size, self.dropout)
        self.verb_embedder.load_state_dict(save_dict['verb_embedder'])
        for i in range(self.max_inputs_allowed):
            self.modules[i].load_state_dict(save_dict[i])

    def set_eval(self):
        self.verb_embedder.eval()
        for i, m in self.modules.items():
            m.eval()

    def set_train(self):
        self.verb_embedder.train()
        for i, m in self.modules.items():
            m.train()
