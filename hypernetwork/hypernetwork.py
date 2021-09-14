import torch
import torch.nn as nn


class myLinear(nn.Linear):
    def __init__(self, *args):
        super(myLinear, self).__init__(*args)


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
            start = end
            end += self.param_sizes[param_id]
            del layer.weight
            w_id = torch.torch.LongTensor(range(start, end)).to(self.device)
            layer.weight = self.weights.index_select(dim=1, index=w_id).view(self.param_shapes[param_id])

            start = end
            end += self.param_sizes[param_id + 1]
            del layer.bias
            w_id = torch.torch.LongTensor(range(start, end)).to(self.device)
            layer.bias = self.weights.index_select(dim=1, index=w_id).view(self.param_shapes[param_id + 1])
            param_id += 2

    def forward(self, x):
        return self.dest_net.forward(x)

    def eval(self):
        self.model.eval()
        self.dest_net.eval()

    def train(self):
        self.model.train()
        self.dest_net.train()
