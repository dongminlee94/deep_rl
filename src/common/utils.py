# -*- coding: utf-8 -*-

"""
Utility functions used in reinforcement learning algorithms
"""

import importlib
import inspect
from typing import Type

import gym
import numpy as np
import torch

from src.algorithms.dqn import DQN
from src.common.networks import MLP


def setup_seed(env: gym.Env, seed: int) -> None:
    """
    Set up a random seed
    """
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_class(module_path: str) -> Type[DQN]:
    """
    Set up a class using module path
    """
    module = importlib.import_module(module_path)

    module_class = []
    for cur_module in inspect.getmembers(module, inspect.isclass):
        if cur_module[1].__module__ == module_path:
            module_class.append(cur_module[0])

    class_object = getattr(module, module_class[0])
    return class_object


def hard_target_update(main: MLP, target: MLP) -> None:
    """
    Update completely from main network parameters to target network parameters
    """
    target.load_state_dict(main.state_dict())


def soft_target_update(main: MLP, target: MLP, tau=0.005) -> None:
    """
    Update partially from main network parameters to target network parameters by tau
    """
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
