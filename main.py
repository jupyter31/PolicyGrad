import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import unittest
from src.log import join, read_config, plot_combined
from src.model import PolicyGradient

import random

parser = argparse.ArgumentParser()
parser.add_argument("--config_filename", required=False, type=str)
parser.add_argument("--plot_config_filename", required=False, type=str)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.config_filename is not None:
        config = read_config(args.config_filename)

        for seed in config["env"]["seed"]:
            torch.random.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            env = gym.make(config["env"]["env_name"])

            # train model
            model = PolicyGradient(env, config, seed)
            model.run()
    else:
        print("Skipping model training as no config provided.")

    if args.plot_config_filename is not None:
        config = read_config(args.plot_config_filename)

        for env in config.keys():
            gym_env_name = config[env]["env_name"]

            all_results = {"Baseline": [], "No baseline": []}
            for seed in config[env]["seed"]:
                baseline_directory = "./results/{}-{}-baseline/".format(
                    gym_env_name, seed
                )
                no_baseline_directory = "./results/{}-{}-no-baseline/".format(
                    gym_env_name, seed
                )
                if not os.path.isdir(no_baseline_directory):
                    sys.exit(
                        "{} was not found. Please ensure you have generated results for this environment, seed and baseline combination".format(
                            no_baseline_directory
                        )
                    )
                if not os.path.isdir(baseline_directory):
                    sys.exit(
                        "{} was not found. Please ensure you have generated results for this environment, seed and baseline combination".format(
                            baseline_directory
                        )
                    )
                all_results["Baseline"].append(
                    np.load(baseline_directory + "scores.npy")
                )
                all_results["No baseline"].append(
                    np.load(no_baseline_directory + "scores.npy")
                )

            plt.figure()
            plt.title(gym_env_name)
            plt.xlabel("Iteration")
            for name, results in all_results.items():
                plot_combined(name, results)
            plt.legend()
            plt.savefig("./results/{}".format(gym_env_name), bbox_inches="tight")

    else:
        print("Skipping generating plot of multiple seeds as no config provided")
