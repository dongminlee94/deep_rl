import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64), activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        input_size = self.obs_dim
        for next_size in self.hidden_sizes:
            fc = nn.Linear(input_size, next_size)
            input_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        self.output_layer = nn.Linear(input_size, self.act_dim)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x


class CategoricalDist(MLP):
    def forward(self, x):
        x = super(CategoricalDist, self).forward(x)
        pi = F.softmax(x, dim=-1)

        dist = Categorical(pi)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_pi, entropy, pi