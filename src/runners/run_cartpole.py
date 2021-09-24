# -*- coding: utf-8 -*-

"""
Module that runs reinforcement learning algorithms on CartPole environment
"""

import datetime
import os
from typing import Any, Dict, Tuple, Type

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from src.algorithms.dqn import DQN
from src.common.utils import setup_seed


class CartPoleRunner:  # pylint: disable=too-many-instance-attributes
    """
    CartPole runner class that runs an algorithms on CartPole environment
    """

    def __init__(
        self,
        algo_class: Type[DQN],
        device: torch.device,
        algo_config: Dict[str, Any],
        **main_config,
    ) -> None:
        self.device = device
        self.seed: int = main_config["seed"]
        self.exp_name: str = main_config["exp_name"]
        self.file_name: str = main_config["file_name"]

        self.env = gym.make("CartPole-v1")
        self.observ_dim: int = self.env.observation_space.shape[0]
        self.action_dim: int = self.env.action_space.n
        self.max_steps: int = self.env._max_episode_steps

        self.algorithm = algo_class(
            observ_dim=self.observ_dim,
            action_num=self.action_dim,
            device=self.device,
            **algo_config,
        )

        if not self.file_name:
            self.file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.writer = SummaryWriter(log_dir=os.path.join("results", self.exp_name, self.file_name))

    def rollout(self, eval_mode: bool = False) -> Tuple[int, float]:
        """
        Rollout an episode up to maximum step length or done condition
        """
        step_number = 0
        total_reward = 0.0

        obs = self.env.reset()
        done = False

        # Keep interacting until the agent
        #   1) reaches a terminal state or
        #   2) satisfies the done condition
        while not (done or step_number == self.max_steps):
            if eval_mode:
                pass
            else:
                # Take an action and output next observation, reward, and done condition
                action = self.algorithm.select_action(torch.Tensor(obs).to(self.device))
                next_obs, reward, done, _ = self.env.step(action)

                # Collect experience (s, a, r, s')
                self.algorithm.collect_experience(
                    obs=obs, action=action, reward=reward, next_obs=next_obs, done=done
                )

                # Train network(s)
                self.algorithm.train_network()

            total_reward += reward
            step_number += 1
            obs = next_obs
        return step_number, total_reward

    def run(self):
        """
        Run an algorithm on CartPole environment
        """
        setup_seed(env=self.env, seed=self.seed)
