import torch
import torch.nn as nn


class FC_Hypernetwork(nn.Module):
    def __init__(self, dim, dest_net, device):
        super(FC_Hypernetwork, self).__init__()
        self.param_shapes = []
        self.param_sizes = []
        self.dim = dim
        self.dest_net = dest_net
        self.device = device
        self.model = None
        self.weights = None
        self.init_model()

    def init_model(self):
        for param in self.dest_net.parameters():
            self.param_shapes.append(param.shape)
            self.param_sizes.append(param.numel())
        output_size = sum(self.param_sizes)
        input_dims = [self.dim, 512]
        output_dims = [512, output_size]
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dims[0], output_dims[0]),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(input_dims[1], output_dims[1])).to(self.device)

    def set_hyper_param(self, hyper_param):
        weights = self.model.forward(hyper_param)
        start = 0
        end = start
        param_id = 0
        for layer in self.dest_net:
            if not isinstance(layer, torch.nn.Linear):
                continue
            start = end
            end += self.param_sizes[param_id]
            del layer.weight
            layer.weight = weights.index_select(
                dim=1,
                index=torch.torch.LongTensor(range(start, end)).to(self.device)).view(self.param_shapes[param_id])

            start = end
            end += self.param_sizes[param_id + 1]
            del layer.bias
            layer.bias = weights.index_select(
                dim=1,
                index=torch.torch.LongTensor(range(start, end)).to(self.device)).view(self.param_shapes[param_id + 1])
            param_id += 2

    def forward(self, x):
        return self.dest_net.forward(x)

    def eval(self):
        self.model.eval()
        self.dest_net.eval()

    def train(self):
        self.model.train()
        self.dest_net.train()

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    def load_state_dict(self, d):
        #         self.model.load_state_dict({'0.weight': d['model.0.weight'], '2.weight': d['model.2.weight'], '0.bias': d['model.0.bias'], '2.bias': d['model.2.bias']})
        self.model.load_state_dict(d)
        self.model = self.model.to(self.device)

    def state_dict(self):
        return self.model.state_dict()
