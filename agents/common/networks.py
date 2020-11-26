import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def identity(x):
    """Return input without any change."""
    return x


"""
DQN, DDQN, A2C critic, VPG critic, TRPO critic, PPO critic, DDPG actor, TD3 actor
"""
class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 output_limit=1.0,
                 hidden_sizes=(64,64), 
                 activation=F.relu, 
                 output_activation=identity,
                 use_output_layer=True,
                 use_actor=False,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.use_actor = use_actor

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        # If the network is used as actor network, make sure output is in correct range
        x = x * self.output_limit if self.use_actor else x   
        return x


"""
A2C actor
"""
class CategoricalPolicy(MLP):
    def forward(self, x):
        x = super(CategoricalPolicy, self).forward(x)
        pi = F.softmax(x, dim=-1)

        dist = Categorical(pi)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        return action, pi, log_pi


"""
DDPG critic, TD3 critic, SAC qf
"""
class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x,a], dim=-1)
        return super(FlattenMLP, self).forward(q)


"""
VPG actor, TRPO actor, PPO actor
"""
class GaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 output_limit=1.0,
                 hidden_sizes=(64,64),
                 activation=torch.tanh,
    ):
        super(GaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        self.output_limit = output_limit
        self.log_std = np.ones(output_size, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))

    def forward(self, x, pi=None, use_pi=True):
        mu = super(GaussianPolicy, self).forward(x)
        std = torch.exp(self.log_std)

        dist = Normal(mu, std)
        if use_pi:
            pi = dist.sample()
        log_pi = dist.log_prob(pi).sum(dim=-1)

        # Make sure outputs are in correct range
        mu = mu * self.output_limit
        pi = pi * self.output_limit
        return mu, std, pi, log_pi


"""
SAC actor
"""
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ReparamGaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size,
                 output_limit=1.0,
                 hidden_sizes=(64,64),
                 activation=F.relu,
    ):
        super(ReparamGaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]
        self.output_limit = output_limit

        # Set output layers
        self.mu_layer = nn.Linear(in_size, output_size)
        self.log_std_layer = nn.Linear(in_size, output_size)        

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x)*clip_up + (l - x)*clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        log_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6), dim=-1)
        return mu, pi, log_pi

    def forward(self, x):
        x = super(ReparamGaussianPolicy, self).forward(x)
        
        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        
        # https://pytorch.org/docs/stable/distributions.html#normal
        dist = Normal(mu, std)
        pi = dist.rsample() # Reparameterization trick (mean + std * N(0,1))
        log_pi = dist.log_prob(pi).sum(dim=-1)
        mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
        
        # Make sure outputs are in correct range
        mu = mu * self.output_limit
        pi = pi * self.output_limit
        return mu, pi, log_pi