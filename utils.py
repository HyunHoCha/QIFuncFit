import torch
import torch.nn as nn
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self, num_units):
        super(SimpleModel, self).__init__()
        self.a = nn.Parameter(torch.randn(num_units, 1))  # a_i
        self.b = nn.Parameter(torch.randn(num_units, 1))  # b_i
        self.c = nn.Parameter(torch.randn(num_units, 1))  # c_i

        # self.a_scale = nn.Parameter(torch.randn(1))
        # self.b_scale = nn.Parameter(torch.randn(1))
        # self.c_scale = nn.Parameter(torch.randn(1))

        # self.activation = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # a_scaled = torch.sigmoid(self.a_scale) * self.a / torch.linalg.norm(self.a)
        # b_scaled = torch.sigmoid(self.b_scale) * self.b / torch.linalg.norm(self.b)
        # c_scaled = torch.sigmoid(self.c_scale) * self.c / torch.linalg.norm(self.c)
        # z = self.activation(x @ a_scaled.T + b_scaled.T)
        # y = z @ c_scaled

        z = self.activation(x @ self.a.T + self.b.T) / (torch.linalg.norm(self.a) + torch.linalg.norm(self.b))
        y = z @ self.c / torch.linalg.norm(self.c)

        # z = self.activation(x @ self.a.T + self.b.T)
        # y = z @ self.c

        return y
