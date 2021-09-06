
class ProcessingException(Exception):
    def __init__(self, *args, **kwargs):
        super(Exception, self).__init__(*args, **kwargs)


class ActionModuleWrapper(object):

    def __init__(self, action_module_facade):
        self.param = None
        self.inputs = []
        self.input_count = 0
        self.module = action_module_facade

    def forward(self, code):
        return self.module.forward(self.param, self.inputs, code)

    def add_input(self, input):
        if len(self.inputs) > 0 and len(self.inputs[-1]) == 1:
            self.inputs[-1].append(input)
        else:
            empty_emb = self.module.modules[1].embedder.embed([' '], [' '])[1]
            self.inputs.append([empty_emb, input])

    def add_preposition(self, prep):
        prep_emb = self.module.modules[1].embedder.embed([prep], [' '])[1]
        self.inputs.append([prep_emb])


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
