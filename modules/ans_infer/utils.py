import torch.nn as torch_nn
import torch

from modules.utils import transpose


class AttentivePooling(torch_nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = torch_nn.Linear(dim, 1)

    def forward(self, X):
        # X: [batch, x, dim]

        X_      = torch.tanh(X)
        alpha   = torch.softmax(self.linear(X_), dim=1)
        # alpha: [batch, x, 1]
        r       = torch.bmm(transpose(X), alpha)
        # r: [batch, dim, 1]

        return r.squeeze(-1)
