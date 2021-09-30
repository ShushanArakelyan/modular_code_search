import torch

import codebert_embedder as embedder
from hypernetwork.hypernetwork import FC_Hypernetwork
from layout_assembly.action_v1 import ActionModule_v1
from layout_assembly.utils import ProcessingException, FC2, FC2_normalized


# Hypernetwork, where the MLPs are parametrized by the verb
class ActionModule_v2(ActionModule_v1):
    def set_hyper_param(self, verb_embedding):
        self.model1.set_hyper_param(verb_embedding)
        return verb_embedding


class ActionModule_v2_one_input(ActionModule_v2):
    def init_networks(self):
        # outputs scores
        hidden_input_dims = [embedder.dim * 2 + 1, 128]
        hidden_output_dims = [128, 1]
        if self.normalized:
            dest_net = FC2_normalized(hidden_input_dims, hidden_output_dims)
        else:
            dest_net = FC2(hidden_input_dims, hidden_output_dims)
        self.model1 = FC_Hypernetwork(embedder.dim, dest_net, self.device)

        # outputs an embedding
        hidden_input_dims = [embedder.dim * 2 + embedder.max_seq_length, 128]
        hidden_output_dims = [128, embedder.dim]
        self.model2 = FC2(hidden_input_dims, hidden_output_dims).to(self.device)

        if eval:
            self.eval()

    def forward(self, _, arg1, __, precomputed_embeddings):
        prep_embedding, scores = arg1[0]
        if isinstance(scores, tuple):
            prep_embedding = (scores[0] + prep_embedding) / 2
            scores = scores[1]
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(dim=1)

        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        self.set_hyper_param(verb_embedding)
        scores_out = self.model1.forward(torch.cat(
            (prep_embedding.repeat(embedder.max_seq_length, 1),
             code_embeddings,
             scores), dim=1))
        emb_out = self.model2.forward(
            torch.cat((verb_embedding, prep_embedding, scores_out.squeeze().unsqueeze(dim=0)), dim=1))
        return emb_out, scores_out


class ActionModule_v2_two_inputs(ActionModule_v2):
    def init_networks(self):
        # outputs scores
        hidden_input_dims = [embedder.dim * 3 + 2, 128]
        hidden_output_dims = [128, 1]
        if self.normalized:
            dest_net = FC2_normalized(hidden_input_dims, hidden_output_dims)
        else:
            dest_net = FC2(hidden_input_dims, hidden_output_dims)
        self.model1 = FC_Hypernetwork(embedder.dim, dest_net, self.device)

        # outputs an embedding
        hidden_input_dims = [embedder.dim * 3 + embedder.max_seq_length, 128]
        hidden_output_dims = [128, embedder.dim]
        self.model2 = FC2(hidden_input_dims, hidden_output_dims).to(self.device)

        if eval:
            self.eval()

    def forward(self, _, args, __, precomputed_embeddings):
        arg1, arg2 = args
        prep1_embedding, scores1 = arg1
        if isinstance(scores1, tuple):
            prep1_embedding = (scores1[0] + prep1_embedding) / 2
            scores1 = scores1[1]
        if len(scores1.shape) == 1:
            scores1 = scores1.unsqueeze(dim=1)
        prep2_embedding, scores2 = arg2
        if isinstance(scores2, tuple):
            prep2_embedding = (scores2[0] + prep2_embedding) / 2
            #             print(prep2_embedding)
            scores2 = scores2[1]
        #             print(verb)
        if len(scores2.shape) == 1:
            scores2 = scores2.unsqueeze(dim=1)
        if precomputed_embeddings is None:
            raise ProcessingException()
        verb_embedding, code_embeddings = precomputed_embeddings
        self.set_hyper_param(verb_embedding)
        scores_out = self.model1.forward(torch.cat(
            (prep1_embedding.repeat(embedder.max_seq_length, 1),
             prep2_embedding.repeat(embedder.max_seq_length, 1),
             code_embeddings,
             scores1,
             scores2), dim=1))
        emb_out = self.model2.forward(torch.cat((verb_embedding,
                                                 prep1_embedding,
                                                 prep2_embedding,
                                                 scores_out.squeeze().unsqueeze(dim=0)), dim=1))
        return emb_out, scores_out
