class ActionModule(object):
    pass


class ScoringModule(object):
    def __init__(self):
        self.inputs = None

    def forward(self, value):
        return value
