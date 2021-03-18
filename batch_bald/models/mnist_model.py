from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor
import torch

from batch_bald.models import mc_dropout


class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = mc_dropout.MCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = mc_dropout.MCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)

        return input

    def get_params(self):
        W1 = self.conv1.weight.flatten()
        b1 = self.conv1.bias
        W2 = self.conv2.weight.flatten()
        b2 = self.conv2.bias
        W3 = self.fc1.weight.flatten()
        b3 = self.fc1.bias
        W4 = self.fc2.weight.flatten()
        b4 = self.fc2.bias
        weight = torch.cat([W1,W2,W3,W4])
        bias = torch.cat([b1,b2,b3,b4])
        return [weight,bias]

