import torch

from scoring.embedder import Embedder
from layout_assembly.utils import ProcessingException
from action_v1 import ActionModule_v1_one_input, ActionModule_v1_two_inputs



class ActionModuleFacade_v1:
    def __init__(self, device):
        self.device = device
        self.one_input_module = ActionModule_v1_one_input(device)
        self.two_inputs_module = ActionModule_v1_two_inputs(device)
        self.modules = {1: self.one_input_module, 2: self.two_inputs_module}

    def forward(self, verb, inputs, code):
        self.num_inputs = len(inputs)
        if self.num_inputs > 2:
            raise ProcessingException()
        #         assert num_inputs <= 2, f'Too many inputs, handling {num_inputs} inputs is not implemented for ActionModule_v1'
        module = self.modules[self.num_inputs]
        self.emb, self.pred = module.forward(verb, inputs, code)
        return self.emb, self.pred

    def parameters(self):
        return self.one_input_module.parameters() + self.two_inputs_module.parameters()
    
    def save_to_checkpoint(self, checkpoint):
        state_dict = {'one_input': self.one_input_module.state_dict(), 
                     'two_inputs': self.two_inputs_module.state_dict()}
        torch.save(state_dict, checkpoint)
    
    def load_from_checkpoint(self, checkpoint):
        models = torch.load(checkpoint, map_location=self.device)
        self.one_input_module.load_state_dict(models['one_input'])
        self.one_input_module = self.one_input_module.to(self.device)
        self.two_inputs_module.load_state_dict(models['two_inputs'])
        self.two_inputs_module.model = self.two_inputs_module.model.to(self.device)
      

    
    
class ScoringModule:
    def __init__(self, device, checkpoint=None):
        self.embedder = Embedder(device, model_eval=False)
        self.scorer = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(device)

        # TODO: not sure about these
        # self.op = torch.optim.Adam(list(self.scorer.parameters()) + list(self.embedder.model.parameters()), lr=1e-8)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

        self.device = device
        if checkpoint:
            models = torch.load(checkpoint, map_location=device)
            self.scorer.load_state_dict(models['scorer'])
            self.scorer = self.scorer.to(device)
            self.embedder.model.load_state_dict(models['embedder'])
            self.embedder.model = self.embedder.model.to(device)
            # self.op.load_state_dict(models['optimizer'])

    def forward(self, query, sample):
        with torch.no_grad():
            _, code, static_tags, regex_tags, ccg_parse = sample
            query_embedder_out = self.embedder.embed(query, [' '])
            if query_embedder_out is None:
                raise ProcessingException()
            self.word_token_id_mapping, self.word_token_embeddings, _, __, ___, ____, self.cls_token_embedding = query_embedder_out
            code_embedder_out = self.embedder.embed([' '], code)
            if code_embedder_out is None:
                raise ProcessingException()
            _, __, self.code_token_id_mapping, self.code_embedding, ___, self.truncated_code_tokens, ____ = code_embedder_out
            if self.word_token_id_mapping.size == 0 or self.code_token_id_mapping.size == 0:
                raise ProcessingException()

            tiled_emb = self.cls_token_embedding.repeat(len(self.truncated_code_tokens), 1)
            forward_input = torch.cat((tiled_emb, self.code_embedding), dim=1)
            token_count = max(self.code_token_id_mapping[-1])
            #         print('in scoring: ', token_count, ', query: ', query)
            scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze()[:token_count]
            return scorer_out

    def compute_loss(self):
        pass
