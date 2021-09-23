# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) algorithm
"""

import random

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
        observ_dim: int,
        action_num: int,
        device: torch.device,
        **config,
    ) -> None:

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
        self.decaying_epsilon_period: int = config["decaying_epsilon_period"]
        self.target_update_period: int = config["target_update_period"]
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

        # Set up replay buffer
        self.replay_buffer = ReplayBuffer(
            observ_dim=self.observ_dim,
            action_dim=self.action_num,
            max_buffer_size=self.max_buffer_size,
            batch_size=self.batch_size,
            device=self.device,
        )

        # Epsilon setup
        self.epsilon = self.initial_epsilon
        self.decaying_epsilon_ratio = self.final_epsilon / (
            self.initial_epsilon * self.decaying_epsilon_period
        )

    def select_action(self, obs: torch.Tensor) -> int:
        """
        Select an action given observation
        """
        if random.random() <= self.epsilon:
            # Select a random action with epsilon probability
            action = random.randint(0, self.action_num - 1)
        else:
            # Select the action with highest Q-value at current state
            action = self.q_network(obs).argmax().detach().cpu().numpy()

        # Decay epsilon as much as ratio
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.decaying_epsilon_ratio
        return action

    def collect_experience(  # pylint: disable=too-many-arguments
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """
        Collect experience (s, a, r, s') and add them to replay buffer
        """
        self.replay_buffer.store(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def train_network(self) -> None:
        """
        Train Q-network
        """
        # Start training when the number of experience is greater than batch_size
        if self.replay_buffer.ptr > self.batch_size:
            batch = self.replay_buffer.sample()
            cur_obs = batch["cur_obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_obs = batch["next_obs"]
            dones = batch["dones"]

            # Get Q-value
            q_value: torch.Tensor = self.q_network(cur_obs)
            q_value = q_value.gather(1, actions.long())

            # Get target Q-value
            if not self.is_double_dqn:  # DQN
                target_next_q_value: torch.Tensor = self.target_q_network(next_obs)
                target_next_q_value = target_next_q_value.max(dim=-1)[0]
            else:  # Double DQN
                next_q_value: torch.Tensor = self.q_network(next_obs)
                target_next_q_value = self.target_q_network(next_obs)
                target_next_q_value = target_next_q_value.gather(1, next_q_value.max(dim=-1)[1])
            target_q_value = rewards + self.gamma * (1 - dones) * target_next_q_value

            # Update main Q-network parameters
            q_loss = F.mse_loss(q_value, target_q_value.detach())
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            # Synchronize target network parameters 𝜃‾ as 𝜃 every C steps
            if self.replay_buffer.ptr % self.target_update_period == 0:
                hard_target_update(main=self.q_network, target=self.target_q_network)
