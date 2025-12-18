"""
Trains PPO baseline agent.
"""

import argparse
import logging
import os
from typing import Any, Dict, Tuple

import grid2op
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from src.ra_agents.RAFeatureExtractor import RLlibGNNModel
from src.rl4pnc.experiments.utils import run_training
from src.rl4pnc.experiments.yaml import load_config
from src.rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from src.rl4pnc.multi_agent.policy import (
    DoNothingPolicy,
    SelectAgentPolicy,
)

REPORT_END = False
ModelCatalog.register_custom_model("gnn_model", RLlibGNNModel)


def setup_config(workdir_path: str, input_path: str, seed: int = None, opponent=False, model_type: str="MLP") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads the JSON as configs and sets it up for training.
    """
    # load base PPO configs and load in hyperparameters
    # Access the parsed arguments
    os.chdir(workdir_path)
    config_path = os.path.join(workdir_path, input_path)
    ppo_config = ppo.PPOConfig().to_dict()
    ppo_config["_disable_preprocessor_api"] = True
    custom_config = load_config(config_path)
    if seed:
        print(f"Running experiment with seed {seed}.")
        custom_config["debugging"]["seed"] = seed
        custom_config["environment"]["env_config"]["seed"] = seed

    # Set observation space based on model_type
    print(f"Using model type: {model_type}")
    if model_type == "GNN":
        custom_config["environment"]["env_config"]["observation_space"] = "BusConnectivityGraphObsSpace"
    else:  # MLP
        custom_config["environment"]["env_config"]["observation_space"] = "BoxGymObsSpace"

    for key in custom_config.keys():
        if key != "setup":
            ppo_config.update(custom_config[key])
    if opponent:
        print("Train with opponent.")
        opponent_path = os.path.join(workdir_path, f"configs/{ppo_config['env_config']['env_name'].replace('_train', '')}/opponent.yaml")
        opponent_kwargs = load_config(opponent_path)
    else:
        # Get kwargs for no opponent
        print("Train without opponent.")
        opponent_kwargs = grid2op.Opponent.get_kwargs_no_opponent()
    ppo_config["env_config"]["grid2op_kwargs"].update(opponent_kwargs)
    ppo_config["evaluation_config"]["env_config"]["grid2op_kwargs"].update(opponent_kwargs)
    # Set eval duration equal to N available validation episodes
    ppo_config["evaluation_duration"] = len(
        os.listdir(os.path.join(
            f"{grid2op.get_current_local_dir()}",
            ppo_config["evaluation_config"]["env_config"]["env_name"],
            "chronics")
        ))
    change_workdir(workdir_path, ppo_config["env_config"]["env_name"])
    # ppo_config["env_config"]["lib_dir"] = os.path.join(workdir_path, ppo_config["env_config"]["lib_dir"])

    policies = {
        "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
            policy_class=SelectAgentPolicy,
            config=(
                AlgorithmConfig()
                .training(
                    model={
                        "custom_model_config": {
                            "rho_threshold": custom_config["environment"]["env_config"][
                                "rho_threshold"
                            ]
                        }
                    },
                )
                .rollouts(preprocessor_pref=None)
            ),
        ),
        "reinforcement_learning_policy": PolicySpec(config={}), # configured in custom config already
        "do_nothing_policy": PolicySpec(  # performs do-nothing action
            policy_class=DoNothingPolicy,
            config=(AlgorithmConfig()),
        ),
    }

    # load environment and agents manually
    ppo_config.update({"policies": policies})
    ppo_config.update({"env": CustomizedGrid2OpEnvironment})
    ppo_config.update({"trial_info": "trial_id"})
    ppo_config.update({"my_log_level": custom_config["setup"]["my_log_level"]})

    return ppo_config, custom_config


def change_workdir(workdir: str, env_name: str) -> None:
    # Change grid2op path if this exists
    env_path = os.path.join(workdir, f"data_grid2op/{env_name}")
    if os.path.exists(env_path):
        grid2op_data_dir = os.path.join(workdir, "data_grid2op")
        grid2op.change_local_dir(grid2op_data_dir)
    else:
        grid2op.change_local_dir(os.path.expanduser("~/data_grid2op"))
    print(f"Environment data location used is: {grid2op.get_current_local_dir()}")
    # Change dir for RLlib ray_results output and disable the default output
    # os.environ["DEFAULT_STORAGE_PATH"] = os.path.join(workdir, f"runs/{env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default="./configs/l2rpn_case14_sandbox/ppo_baseline.yaml", # "./configs/rte_case5_example/ppo_baseline.yaml", #"./configs/l2rpn_icaps_2021_small/ppo_baseline.yaml",  #
        help="Path to the configs file.",
    )
    parser.add_argument(
        "-wd",
        "--workdir",
        type=str,
        default=".",
        help="path do store results.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Seed of the experiment",
    )
    parser.add_argument(
        "-j",
        "--job_id",
        type=str,
        default="TEsTING",
        help="job_id of this trial, this way each trial gets an extra unique identifier.",
    )
    parser.add_argument(
        "-o",
        "--opponent",
        # default=True,
        action='store_true',
        help="Train on environment with opponent.",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        default="MLP",
        help="Model type to use for RL policy. (MLP, GNN, or RAGNN)",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.file_path:
        _ppo_config, _custom_config = setup_config(args.workdir, args.file_path, seed=args.seed, opponent=args.opponent, model_type=args.model_type)
        _result_grid = run_training(_ppo_config, _custom_config["setup"], args.job_id)
    else:
        parser.print_help()
        logging.error("\nError: --file_path is required to specify configs location.")
