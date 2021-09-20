# -*- coding: utf-8 -*-

"""
Module for deep neural networks used by RL agents
"""

from typing import Callable

import torch
from torch.nn import Linear, Module, ModuleList


class MLP(Module):
    """
    Multi-layer Perceptron (MLP) network class
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_num: int,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
        use_output_activation: bool = False,
        use_output_layer: bool = True,
        is_actor: bool = False,
        init_w: float = 0.003,
    ) -> None:
        super().__init__()

        self.activation_function = activation_function
        self.is_actor = is_actor

        # Set up hidden layers
        self.fc_layers = ModuleList()
        hidden_layers = [hidden_dim] * hidden_num
        in_layer = input_dim
        for i, next_layer in enumerate(hidden_layers):
            fc_layer = Linear(in_layer, next_layer)
            self.__setattr__("fc_layer{}".format(i), fc_layer)
            self.fc_layers.append(fc_layer)
            in_layer = next_layer

        # Set up output activation
        self.output_activation = torch.tanh if use_output_activation else lambda x: x

        # Set up output layers
        self.output_layer = Linear(in_layer, output_dim) if use_output_layer else lambda x: x
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get output over the network when input is given"""
        for fc_layer in self.fc_layers:
            x = self.activation_function(fc_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x
