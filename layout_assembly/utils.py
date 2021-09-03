class ActionModuleWrapper(object):

    def __init__(self, action_module):
        self.param = None
        self.inputs = []
        self.input_count = 0
        self.module = action_module

    def forward(self, code):
        # use self.module to perform forward
        out = "action: " + self.param
        i = 0
        while i < len(self.inputs):
            out += ' ' + self.inputs[i]
            i += 1
        return out

    def add_input(self, input):
        if len(self.inputs) > 0 and len(self.inputs[-1]) == 1:
            self.inputs[-1].append(input)
        else:
            self.inputs.append([None, input])

    def add_preposition(self, prep):
        self.inputs.append([prep])


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
            out += '({1}, {2}), '.format(self.inputs[i][0], self.inputs[i][1])
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


class TestScoringModuleWrapper(object):

    def __init__(self, action_module):
        self.param = None
        self.inputs = []
        self.input_count = 0
        self.module = action_module

    def forward(self, value, _):
        return value
