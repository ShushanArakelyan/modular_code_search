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
