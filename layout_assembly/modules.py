import torch

from layout_assembly.action_v1 import ActionModule_v1_one_input, ActionModule_v1_two_inputs
from layout_assembly.action_v2 import ActionModule_v2_one_input, ActionModule_v2_two_inputs
from layout_assembly.utils import ProcessingException
import codebert_embedder as embedder


class ActionModuleFacade_v1:
    def __init__(self, device, checkpoint=None, eval=False):
        self.device = device
        self.one_input_module = ActionModule_v1_one_input(device, eval)
        self.two_inputs_module = ActionModule_v1_two_inputs(device, eval)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}
        if checkpoint:
            self.load_from_checkpoint(checkpoint)

    def forward(self, verb, inputs, code):
        num_inputs = len(inputs)
        if num_inputs > 2:
            raise ProcessingException()
        module = self.modules[num_inputs]
        emb, pred = module.forward(verb, inputs, code)
        return emb, pred

    def parameters(self):
        return self.one_input_module.parameters() + self.two_inputs_module.parameters()

    def save_to_checkpoint(self, checkpoint):
        state_dict = {'one_input': self.one_input_module.state_dict(),
                      'two_inputs': self.two_inputs_module.state_dict()}
        torch.save(state_dict, checkpoint)

    def load_from_checkpoint(self, checkpoint):
        models = torch.load(checkpoint, map_location=self.device)
        self.one_input_module.load_state_dict(models['one_input'])
        self.two_inputs_module.load_state_dict(models['two_inputs'])


class ActionModuleFacade_v2(ActionModuleFacade_v1):
    def __init__(self, device, checkpoint=None, eval=False):
        ActionModuleFacade_v1.__init__(self, device)
        self.one_input_module = ActionModule_v2_one_input(device, eval)
        self.two_inputs_module = ActionModule_v2_two_inputs(device, eval)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}
        if checkpoint:
            self.load_from_checkpoint(checkpoint)


class ScoringModule:
    def __init__(self, device, checkpoint=None, eval=True):
        if not embedder.initialized:
            embedder.init_model(device)
        self.scorer = torch.nn.Sequential(torch.nn.Linear(embedder.dim * 2, embedder.dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(embedder.dim, 1)).to(device)
        if eval:
            self.scorer.eval()
        self.device = device
        if checkpoint:
            models = torch.load(checkpoint, map_location=device)
            self.scorer.load_state_dict(models['scorer'])
            self.scorer = self.scorer.to(device)
            embedder.model.load_state_dict(models['embedder'])
            embedder.model = embedder.model.to(device)

    def forward(self, query, sample):
        with torch.no_grad():
            _, code, static_tags, regex_tags, ccg_parse = sample
            query_embedder_out = embedder.embed(query, [' '])
            if query_embedder_out is None:
                raise ProcessingException()
            word_token_id_mapping, word_token_embeddings, _, __, ___, ____, cls_token_embedding = query_embedder_out
            code_embedder_out = embedder.embed([' '], code)
            if code_embedder_out is None:
                raise ProcessingException()
            _, _, _, code_embeddings, _, _, _ = code_embedder_out
            padding_size = embedder.max_seq_length - len(code_embeddings)
            code_embeddings = torch.nn.functional.pad(code_embeddings, (0, 0, 0, padding_size), 'constant', 0)
            tiled_emb = cls_token_embedding.repeat(embedder.max_seq_length, 1)
            forward_input = torch.cat((tiled_emb, code_embeddings), dim=1)
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()
            return scorer_out

    def forward_v2(self, query, sample):
        with torch.no_grad():
            _, code, static_tags, regex_tags, ccg_parse = sample
            embedder_out = embedder.embed(query, code)
            if embedder_out is None:
                raise ProcessingException()
            word_token_id_mapping, word_token_embeddings, code_token_id_mapping, code_embedding, _, truncated_code_tokens, cls_token_embedding = embedder_out

            if word_token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
                raise ProcessingException()

            tiled_emb = cls_token_embedding.repeat(len(truncated_code_tokens), 1)
            forward_input = torch.cat((tiled_emb, code_embedding), dim=1)
            token_count = max(code_token_id_mapping[-1])
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()[:token_count]
            # padding_size = embedder.max_seq_length - len(scorer_out)

            print(scorer_out.shape, token_count)
            return scorer_out
