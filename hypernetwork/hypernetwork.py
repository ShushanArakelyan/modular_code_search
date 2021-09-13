import numpy as np
import torch
import torch.nn as nn


class myLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(nn.Linear, self).__init__(*args, **kwargs)

    def set_parameters(self, new_weight, new_bias):
        del self.weight
        self.weight = new_weight
        del self.bias
        self.bias = new_bias


class FC_Hypernetwork(nn.Module):
    def __init__(self, dest_net, device):
        self.param_shapes = []
        self.param_sizes = []
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
        input_dims = [self.dim, 1000]
        output_dims = [1000, output_size]
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dims[0], output_dims[0]),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(input_dims[1], output_dims[1])).to(self.device)

    def set_hyper_param(self, hyper_param):
        self.weights = self.model.forward(hyper_param)
        start = 0
        end = start
        param_id = 0
        for layer in self.dest_net:
            if not isinstance(layer, torch.nn.Linear):
                continue
            end += self.param_sizes[param_id]
            weight_ids = torch.LongTensor(np.arange(start, end)).to(self.device)
            start = end
            end += self.param_sizes[param_id + 1]
            bias_ids = torch.LongTensor(np.arange(start, end)).to(self.device)
            layer.set_parameters(self.weights[weight_ids].view(self.param_shapes.pop()),
                                 self.weights[bias_ids].view(self.param_shapes.pop()))
            param_id += 2

    def forward(self, x):
        return self.dest_net.forward(x)
