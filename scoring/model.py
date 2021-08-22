import torch.nn as nn


class ScoringLayer(nn.Module):
    """Classification layer, where the input is the combined vector of code-query embeddings.
    The task is to predict the whether the embedding are related."""

    def __init__(self, device, dim=768):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(dim * 2, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid()).to(self.device)

    def forward(self, data):
        comb_vector = data.to(self.device)
        output = self.model(comb_vector)
        return output
