# -*- coding: utf-8 -*-

"""
Module for implementation of Deep Q-Network (DQN) algorithm
"""

from typing import Dict

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.common.buffers import ReplayBuffer
from src.common.networks import MLP
from src.common.utils import hard_target_update


class DQN:  # pylint: disable=too-many-instance-attributes
    """
    DQN algorithm class
    """

    def __init__(
        self,
        env: gym.wrappers,
        observ_dim: int,
        action_num: int,
        device: torch.device,
        **config,
    ) -> None:

        self.env = env
        self.observ_dim = observ_dim
        self.action_num = action_num
        self.device = device
        self.hidden_dim: int = config["hidden_dim"]
        self.hidden_num: int = config["hidden_num"]
        self.gamma: float = config["gamma"]
        self.max_buffer_size: int = config["max_buffer_size"]
        self.batch_size: int = config["batch_size"]
        self.initial_epsilon: float = config["initial_epsilon"]
        self.final_epsilon: float = config["final_epsilon"]
        self.epsilon_timesteps: int = config["epsilon_timesteps"]
        self.target_update_freq: int = config["target_update_freq"]
        self.is_double_dqn: bool = config["is_double_dqn"]
        self.eval_mode: bool = config["eval_mode"]

        # Main Q-network
        self.q_network = MLP(
            input_dim=self.observ_dim,
            output_dim=self.action_num,
            hidden_dim=self.hidden_dim,
            hidden_num=self.hidden_num,
            activation_function=F.relu,
        ).to(self.device)

        # Target Q-network
        self.target_q_network = MLP(
            input_dim=self.observ_dim,
            output_dim=self.action_num,
            hidden_dim=self.hidden_dim,
            hidden_num=self.hidden_num,
            activation_function=F.relu,
        ).to(self.device)

        # Initialize target network parameters to match main network parameters
        hard_target_update(main=self.q_network, target=self.target_q_network)

        # Create an optimizer
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(
            observ_dim=self.observ_dim,
            action_dim=1,
            max_buffer_size=self.max_buffer_size,
            batch_size=self.batch_size,
            device=self.device,
        )

        # Epsilon setup
        self.epsilon = self.initial_epsilon
        self.epsilon_decaying_ratio = self.final_epsilon / (
            self.initial_epsilon * self.epsilon_timesteps
        )

    def select_action(self, obs: torch.Tensor) -> int:
        """
        Select an action given observation
        """
        if np.random.rand() <= self.epsilon:
            # Select a random action with epsilon probability
            action = np.random.choice(self.action_num)
        else:
            # Select the action with highest Q-value at current state
            action = self.q_network(obs).argmax().detach().cpu().numpy()

        # Decay epsilon
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decaying_ratio
        return action

    def train_network(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Train Q-network
        """
        cur_obs = batch["cur_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Get Q-value given observation
        q_value: torch.Tensor = self.q_network(cur_obs)
        q_value = q_value.gather(1, actions.long()).squeeze(1)

        # Target for Q regression
        if not self.is_double_dqn:  # DQN
            q_target: torch.Tensor = self.target_q_network(next_obs)
        else:  # Double DQN
            next_q_value: torch.Tensor = self.q_network(next_obs)
            q_target = self.target_q_network(next_obs)
            q_target = q_target.gather(1, next_q_value.max(1)[1].unsqueeze(1))
        q_target = rewards + self.gamma * (1 - dones) * q_target.max(1)[0]

        # Update perdiction network parameter
        q_loss = F.mse_loss(q_value, q_target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Synchronize target network parameters 𝜃‾ as 𝜃 every C steps
        if self.replay_buffer.ptr % self.target_update_freq == 0:
            hard_target_update(main=self.q_network, target=self.target_q_network)

        return q_loss.item()
