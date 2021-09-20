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
        self, obs_dim: int, act_dim: int, max_size: int, batch_size: int, device: torch.device
    ) -> None:
        self._cur_obs_buf = np.zeros((max_size, obs_dim))
        self._actions_buf = np.zeros((max_size, act_dim))
        self._rewards_buf = np.zeros((max_size, 1))
        self._next_obs_buf = np.zeros((max_size, obs_dim))
        self._dones_buf = np.zeros((max_size, 1), dtype="uint8")
        self._ptr, self._size = 0, 0
        self._max_size = max_size
        self._batch_size = batch_size
        self._device = device

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
        self._cur_obs_buf[self._ptr] = obs
        self._actions_buf[self._ptr] = action
        self._rewards_buf[self._ptr] = reward
        self._next_obs_buf[self._ptr] = next_obs
        self._dones_buf[self._ptr] = done

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample(self) -> Dict[str, torch.Tensor]:
        """
        Sampling the agent's experience from each buffer
        """
        indices = np.random.randint(low=0, high=self._size, size=self._batch_size)
        return dict(
            cur_obs=torch.Tensor(self._cur_obs_buf[indices]).to(self._device),
            actions=torch.Tensor(self._actions_buf[indices]).to(self._device),
            rewards=torch.Tensor(self._rewards_buf[indices]).to(self._device),
            next_obs=torch.Tensor(self._actions_buf[indices]).to(self._device),
            dones=torch.Tensor(self._dones_buf[indices]).to(self._device),
        )
