# -*- coding: utf-8 -*-

"""
Main module that runs an RL algorithm in a certain environment
"""

import datetime
import importlib
import inspect
import os
from typing import Any, Dict, Tuple, Type

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.algorithms.dqn import DQN
from src.runners.run_cartpole import CartPoleRunner


def setup_classes(env_name: str, algo_name: str) -> Tuple[Type[CartPoleRunner], Type[DQN]]:
    """
    Set up runner class and algorithm class according to environment name and algorithm name
    """
    classes = []
    run_module_path = f"src.runners.run_{env_name}"
    algo_module_path = f"src.algorithms.{algo_name}"

    for module_path in [run_module_path, algo_module_path]:
        module = importlib.import_module(module_path)

        module_class = []
        for cur_module in inspect.getmembers(module, inspect.isclass):
            if cur_module[1].__module__ == module_path:
                module_class.append(cur_module[0])

        classes.append(getattr(module, module_class[0]))
    return classes[0], classes[1]


if __name__ == "__main__":
    # Main configuration setup
    with open(os.path.join("config", "main.yaml"), "r") as file:
        main_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # Runner configuration setup
    with open(
        os.path.join("config", "runners", "run_" + main_config["env_name"] + ".yaml"), "r"
    ) as file:
        runner_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # Algorithm configuration setup
    with open(os.path.join("config", "algorithms", main_config["algo_name"] + ".yaml"), "r") as file:
        algo_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    runner_class, algo_class = setup_classes(
        env_name=main_config["env_name"], algo_name=main_config["algo_name"]
    )

    device: torch.device = (
        torch.device("cuda", index=main_config["gpu_index"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if not main_config["file_name"]:
        file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(
        log_dir=os.path.join("results", main_config["exp_name"], main_config["file_name"])
    )

    algorithm = runner_class(
        algo_class=algo_class,
        device=device,
        writer=writer,
        algo_config=algo_config,
        **runner_config,
    )

    # algorithm.run()
