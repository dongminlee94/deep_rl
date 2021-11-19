# -*- coding: utf-8 -*-

"""
Main module that runs an RL algorithm in a certain environment
"""

import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.main_util import setup_classes, setup_configs

if __name__ == "__main__":
    main_config, runner_config, algo_config = setup_configs()

    # Set up runner class and algorithm class
    runner_class, algo_class = setup_classes(
        env_name=main_config["env_name"], algo_name=main_config["algo_name"]
    )

    device = (
        torch.device("cuda", index=main_config["gpu_index"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if not main_config["file_name"]:
        main_config["file_name"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(
        log_dir=os.path.join(
            "results",
            main_config["env_name"],
            main_config["algo_name"],
            main_config["exp_name"],
            main_config["file_name"],
        )
    )

    algorithm = runner_class(
        algo_class=algo_class,
        device=device,
        writer=writer,
        algo_config=algo_config,
        **runner_config,
    )

    print(
        f"Finish all the settings"
        f' to run {main_config["algo_name"]} algorithm on {main_config["env_name"]} environment\n'
        f'Start iterations of {main_config["env_name"]} runner\n'
        f"==================================================\n"
    )
    algorithm.run()
