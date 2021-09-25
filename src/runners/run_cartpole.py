# -*- coding: utf-8 -*-

"""
Module that runs reinforcement learning algorithms on CartPole environment
"""

import sys
from typing import Any, Dict, Type

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from src.algorithms.dqn import DQN
from src.utils.runner_util import setup_seed


class CartPoleRunner:  # pylint: disable=too-many-instance-attributes
    """
    CartPole runner class that runs an algorithms on CartPole environment
    """

    def __init__(
        self,
        algo_class: Type[DQN],
        device: torch.device,
        writer: SummaryWriter,
        algo_config: Dict[str, Any],
        **main_config,
    ) -> None:
        self.device = device
        self.writer = writer
        self.seed: int = main_config["seed"]
        self.num_iterations: int = main_config["num_iterations"]
        self.eval_interval: int = main_config["eval_interval"]
        self.threshold_return: int = main_config["threshold_return"]
        self.is_only_eval: bool = main_config["is_only_eval"]
        self.use_rendering: bool = main_config["use_rendering"]

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

    def rollout(self, is_eval_mode: bool) -> float:
        """
        Rollout an episode up to maximum step length or done condition
        """
        num_steps = 0
        num_rewards = 0.0

        obs = self.env.reset()
        done = False

        # Keep interacting until the agent
        #   1) reaches a terminal state or
        #   2) satisfies the done condition
        while not (done or num_steps == self.max_steps):
            if self.use_rendering:
                self.env.render()

            # Take an action and output next observation, reward, and done condition
            action = self.algorithm.select_action(torch.Tensor(obs).to(self.device), is_eval_mode)
            next_obs, reward, done, _ = self.env.step(action)

            if not is_eval_mode:
                # Collect experience (s, a, r, s')
                self.algorithm.collect_experience(
                    obs=obs, action=action, reward=reward, next_obs=next_obs, done=done
                )

                # Train the network(s)
                self.algorithm.train_network()

            num_rewards += reward
            num_steps += 1
            obs = next_obs
        return num_rewards

    def run(self) -> None:
        """
        Run an algorithm on CartPole environment
        """
        setup_seed(env=self.env, seed=self.seed)

        num_returns = 0.0
        num_episodes = 0

        for iteration in range(self.num_iterations):
            print(f"===== Iteration {iteration} =====")

            # Perform the training phase, during which the agent learns
            if not self.is_only_eval:
                # Rollout one episode
                episode_return = self.rollout(is_eval_mode=False)

                num_returns += episode_return
                num_episodes += 1
                average_return = num_returns / num_episodes

                # Visualize log information to tensorboard
                self.writer.add_scalar("train/average_return", average_return, iteration)
                self.writer.add_scalar("train/episode_return", episode_return, iteration)
                for key, value in self.algorithm.log_info.items():
                    self.writer.add_scalar("log_info/" + key, value, iteration)

            # Perform the evaluation phase -- no learning
            if (iteration + 1) % self.eval_interval == 0:
                print(f"Start evaluation in iteration {iteration}")
                self.eval(iteration)

    def eval(self, iteration: int) -> None:
        """
        Evaluate network(s) trained with the algorithm
        """
        num_returns = 0.0
        num_episodes = 0

        for _ in range(100):
            # Rollout one episode
            episode_return = self.rollout(is_eval_mode=True)

            num_returns += episode_return
            num_episodes += 1
        average_return = num_returns / num_episodes

        # Visualize log information to tensorboard
        self.writer.add_scalar("eval/average_return", average_return, iteration)
        self.writer.add_scalar("eval/episode_return", episode_return, iteration)

        # Save the trained model
        if average_return >= self.threshold_return:
            print(
                f"\n==================================================\n"
                f"In evaluation phase, the last average return value is {average_return}.\n"
                f"And early stopping condition is {self.threshold_return}.\n"
                f"Therefore, cartpole runner is terminated."
            )
            sys.exit()
