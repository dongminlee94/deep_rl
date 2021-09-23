# -*- coding: utf-8 -*-

"""
Main module that runs an RL algorithm in a certain environment
"""

import os
from typing import Any, Dict

import torch
import yaml

from src.common.utils import setup_class

if __name__ == "__main__":
    # Main configuration setup
    with open(os.path.join("config", "main_config.yaml"), "r") as file:
        main_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # Algorithm configuration setup
    with open(os.path.join("config", main_config["algo_name"] + "_config.yaml"), "r") as file:
        algo_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    run_module_path = f'src.runners.run_{main_config["env_name"]}'
    run_object = setup_class(run_module_path)

    algo_module_path = f'src.algorithms.{main_config["algo_name"]}'
    algo_object = setup_class(algo_module_path)

    device: torch.device = (
        torch.device("cuda", index=main_config["gpu_index"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # algorithm = run_object(
    #     algo_object=algo_object,
    #     device=device,
    #     **algo_config
    # )

    # algorithm.run()
