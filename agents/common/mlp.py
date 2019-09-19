import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from agents.common.utils import identity


"""
DQN, DDQN, A2C critic, DDPG actor, SAC vf
"""
class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(64,64), 
                 activation=F.relu, 
                 output_activation=identity,
                 use_output_layer=True,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer

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
        return x


"""
DDPG critic, SAC qf, SAC_alpha qf
"""
class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x,a], dim=-1)
        return super(FlattenMLP, self).forward(q)


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
        entropy = dist.entropy()
        return action, log_pi, entropy, pi


"""
SAC actor, SAC_alpha actor
"""
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class GaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(256,256),
    ):
        super(GaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]

        # Set output layers
        self.mu_layer = nn.Linear(in_size, output_size)
        self.log_std_layer = nn.Linear(in_size, output_size)

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x)*clip_up + (l - x)*clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, logp_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi**2, l=0., u=1.) + 1e-6), dim=1)
        
    def forward(self, x):
        x = super(GaussianPolicy, self).forward(x)
        
        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        pi = dist.rsample()
        log_pi = dist.log_prob(pi)
        entropy = dist.entropy()

        mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
        return mu, pi, log_pi, entropy