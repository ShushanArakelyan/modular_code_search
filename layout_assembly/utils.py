import torch

import codebert_embedder as embedder


class ProcessingException(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)


class ActionModuleWrapper(object):
    empty_emb = None
    prep_emb_cache = {}

    def __init__(self, action_module_facade):
        self.param = None
        self.inputs = []
        self.input_count = 0
        self.module = action_module_facade

    def forward(self, code, verb_embedding):
        return self.module.forward(self.param, self.inputs, code, verb_embedding)

    def add_input(self, input):
        if len(self.inputs) > 0 and len(self.inputs[-1]) == 1:
            self.inputs[-1].append(input)
        else:
            if ActionModuleWrapper.empty_emb is None:
                with torch.no_grad():
                    ActionModuleWrapper.empty_emb = embedder.embed([' '], [' '])[1]
            self.inputs.append([ActionModuleWrapper.empty_emb, input])

    def add_preposition(self, prep):
        if prep not in ActionModuleWrapper.prep_emb_cache:
            with torch.no_grad():
                ActionModuleWrapper.prep_emb_cache[prep] = embedder.embed([prep], [' '])[1]
        self.inputs.append([ActionModuleWrapper.prep_emb_cache[prep]])

    def set_eval(self):
        self.module.set_eval()

    def set_train(self):
        self.module.set_train()


class TestActionModuleWrapper(object):

    def __init__(self, action_module):
        self.param = None
        self.inputs = []
        self.input_count = 0
        self.module = action_module

    def forward(self, _):
        # use self.module to perform forward
        out = "action(" + self.param
        i = 0
        while i < len(self.inputs):
            out += '({0}, {1}), '.format(self.inputs[i][0], self.inputs[i][1])
            i += 1
        out += ')'
        return out

    def add_input(self, input):
        if len(self.inputs) > 0 and len(self.inputs[-1]) == 1:
            self.inputs[-1].append(input)
        else:
            self.inputs.append([None, input])

    def add_preposition(self, prep):
        self.inputs.append([prep])


class TestScoringModuleWrapper:

    def __init__(self, ):
        pass

    def forward(self, value, _):
        return value

    
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    
    
def FC2(hidden_input_dims, hidden_output_dims):
    return torch.nn.Sequential(torch.nn.Linear(hidden_input_dims[0], hidden_output_dims[0]),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_input_dims[1], hidden_output_dims[1]))


def FC2_normalized(hidden_input_dims, hidden_output_dims, dropout=0):
#     if dropout > 0:
#         return torch.nn.Sequential(torch.nn.Linear(hidden_input_dims[0], hidden_output_dims[0]),
#                            torch.nn.ReLU(),
#                            torch.nn.Dropout(0.1)
#                            torch.nn.Linear(hidden_input_dims[1], hidden_output_dims[1]),
#                            torch.nn.Sigmoid())

    return torch.nn.Sequential(torch.nn.Linear(hidden_input_dims[0], hidden_output_dims[0]),
                               torch.nn.ReLU(),
                               torch.nn.Linear(hidden_input_dims[1], hidden_output_dims[1]),
                               torch.nn.Sigmoid())
