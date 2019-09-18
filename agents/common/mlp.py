import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64,64), activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        self.output_layer = nn.Linear(in_size, self.output_size)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x


class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x,a], dim=-1)
        return super(FlattenMLP, self).forward(q)


class CategoricalDist(MLP):
    def forward(self, x):
        x = super(CategoricalDist, self).forward(x)
        pi = F.softmax(x, dim=-1)

        dist = Categorical(pi)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_pi, entropy, pi