# -*- coding: utf-8 -*-

"""
Module for buffers used in reinforcement learning algorithms
- off-policy: ReplayBuffer
- on-policy: Buffer
"""

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:  # pylint: disable=too-many-instance-attributes
    """
    Replay buffer class that can store the agent's experience
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim: int,
        action_dim: int,
        max_buffer_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.cur_obs_buf = np.zeros((max_buffer_size, observ_dim))
        self.actions_buf = np.zeros((max_buffer_size, action_dim))
        self.rewards_buf = np.zeros((max_buffer_size, 1))
        self.next_obs_buf = np.zeros((max_buffer_size, observ_dim))
        self.dones_buf = np.zeros((max_buffer_size, 1), dtype="uint8")
        self.ptr, self.cur_buffer_size = 0, 0
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.device = device

    def store(  # pylint: disable=too-many-arguments
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """
        Storing the agent's experience to each buffer
        """
        self.cur_obs_buf[self.ptr] = obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.dones_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_buffer_size
        self.cur_buffer_size = min(self.cur_buffer_size + 1, self.max_buffer_size)

    def sample(self) -> Dict[str, torch.Tensor]:
        """
        Sampling the agent's experience from each buffer
        """
        indices = np.random.choice(self.cur_buffer_size, size=self.batch_size, replace=False)
        return dict(
            cur_obs=torch.Tensor(self.cur_obs_buf[indices]).to(self.device),
            actions=torch.Tensor(self.actions_buf[indices]).to(self.device),
            rewards=torch.Tensor(self.rewards_buf[indices]).to(self.device),
            next_obs=torch.Tensor(self.actions_buf[indices]).to(self.device),
            dones=torch.Tensor(self.dones_buf[indices]).to(self.device),
        )
