# -*- coding: utf-8 -*-

"""
The main module that runs an RL algorithm in a certain environment
"""

import datetime
import os
import time
from typing import Dict

import gym
import numpy as np
import pybullet_envs as bullet
import torch
import yaml

if __name__ == "__main__":
    # Main configuration setup
    with open(os.path.join("configs", "main_config.yaml"), "r") as file:
        main_config: Dict[str, bool] = yaml.load(file, Loader=yaml.FullLoader)

    # Environment setup
    if main_config["is_discrete_env"]:  # Discrete environment
        env = gym.make("CartPole-v1")  # 4, 2

        obs_dim: int = env.observation_space.shape[0]
        act_num: int = env.action_space.n
        print(f"Observation dimension: {obs_dim}\n" f"Action number: {act_num}")
    else:  # Continuous environments
        # env = gym.make('Pendulum-v0')  # 3, 1
        # env = gym.make('Hopper-v2')  # 11, 3
        # env = gym.make('HalfCheetah-v2')  # 17, 6
        # env = gym.make('Ant-v2')  # 111, 8
        # env = gym.make('Humanoid-v2')  # 376, 17

        # env = bullet.make('HopperBulletEnv-v0')  # 15, 3
        env = bullet.make("HalfCheetahBulletEnv-v0")  # 26, 6

        obs_dim = env.observation_space.shape[0]
        act_dim: int = env.action_space.shape[0]
        print(f"Observation dimension: {obs_dim}\n" f"Action dimension: {act_dim}")

    print(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print(np.__version__)
    print(time.time())
    print(torch.__version__)
