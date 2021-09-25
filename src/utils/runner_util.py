# -*- coding: utf-8 -*-

"""
Utility functions used in runners
"""

import gym
import numpy as np
import torch


def setup_seed(env: gym.Env, seed: int) -> None:
    """
    Set up a random seed
    """
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
