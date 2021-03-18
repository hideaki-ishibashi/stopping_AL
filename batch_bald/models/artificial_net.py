from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor
import torch

from batch_bald.models import mc_dropout


class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, input_dim,hidden_dim,num_class):
        super().__init__(num_class)
        self.activation = F.relu
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_drop = mc_dropout.MCDropout()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_drop = mc_dropout.MCDropout()
        self.fc4 = nn.Linear(hidden_dim, num_class)

    def mc_forward_impl(self, input: Tensor):
        input = self.activation(self.fc1_drop(self.fc1(input)))
        input = self.activation(self.fc2_drop(self.fc2(input)))
        input = self.activation(self.fc3_drop(self.fc3(input)))
        input = self.fc4(input)
        # input = F.softmax(self.fc4(input),dim=1)

        return input

    def get_params(self):
        W1 = self.fc1.weight.flatten()
        b1 = self.fc1.bias
        W2 = self.fc2.weight.flatten()
        b2 = self.fc2.bias
        W3 = self.fc3.weight.flatten()
        b3 = self.fc3.bias
        W4 = self.fc4.weight.flatten()
        b4 = self.fc4.bias
        weight = torch.cat([W1,W2,W3,W4])
        bias = torch.cat([b1,b2,b3,b4])
        return [weight,bias]
