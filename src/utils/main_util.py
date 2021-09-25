# -*- coding: utf-8 -*-

"""
Utility functions used in main module
"""

import importlib
import inspect
import os
from typing import Any, Dict, Tuple, Type

import yaml

from src.algorithms.dqn import DQN
from src.runners.run_cartpole import CartPoleRunner


def setup_configs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Set up runner and algorithm config according to environment and algorithm name
    """
    # Set up main configuration
    with open(os.path.join("configs", "main.yaml"), "r") as file:
        main_config = yaml.load(file, Loader=yaml.FullLoader)

    # Set up runner and algorithm configuration
    runner_config_path = os.path.join("configs", "runners", "run_" + main_config["env_name"] + ".yaml")
    algo_config_path = os.path.join("configs", "algorithms", main_config["algo_name"] + ".yaml")

    configs = []
    for config_path in [runner_config_path, algo_config_path]:
        with open(config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        configs.append(config)
    return main_config, configs[0], configs[1]


def setup_classes(env_name: str, algo_name: str) -> Tuple[Type[CartPoleRunner], Type[DQN]]:
    """
    Set up runner and algorithm class according to environment and algorithm name
    """
    run_module_path = f"src.runners.run_{env_name}"
    algo_module_path = f"src.algorithms.{algo_name}"

    classes = []
    for module_path in [run_module_path, algo_module_path]:
        module = importlib.import_module(module_path)

        module_class = []
        for cur_module in inspect.getmembers(module, inspect.isclass):
            if cur_module[1].__module__ == module_path:
                module_class.append(cur_module[0])

        classes.append(getattr(module, module_class[0]))
    return classes[0], classes[1]
