import logging
from pathlib import Path
from typing import Optional

import grid2op
import hydra
import torch
from hydra.utils import instantiate
from lightsim2grid import LightSimBackend
from omegaconf import DictConfig
from stable_baselines3 import DQN

from baselines.baseline_agent import BaselineAgent, evaluate_agent
from baselines.topo_policy_stable_dqn import TopoPolicyStableDQN
from common import Grid2OpEnvWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_path_models = Path("data/models/stable-baselines/")
base_path_logs = Path("data/logs/stable-baselines/")
base_path_evaluations = Path("data/evaluations/stable-baselines/")


def train(cfg: DictConfig):
    """
    Creates a model using the setup_method, trains it and saves it.
    """
    dqn = model_setup(cfg)
    try:
        dqn.learn(cfg.train.timesteps, log_interval=10)
    except KeyboardInterrupt:
        answered = False
        while not answered:
            save = input("You interrupted the training. Do you want to safe the model? (Y/N)")
            if save.lower() in ["n", "no"]:
                raise KeyboardInterrupt()
            elif save.lower() in ["y", "yes"]:
                dqn.save(Path(base_path_models.joinpath(cfg.model.name)))
                raise KeyboardInterrupt()

    dqn.save(Path(base_path_models.joinpath(cfg.model.name)))


def evaluate(cfg: DictConfig):
    for dataset in ["train", "test", "val"]:
        env = grid2op.make(f"{cfg.env.env_name}_{dataset}", backend=LightSimBackend())
        evaluate_agent(
            agent=build_agent(cfg, Path(base_path_models.joinpath(cfg.model.name))),
            env=env,
            num_episodes=50, # len(env.chronics_handler.subpaths),  # run all episodes
            path_results=Path(base_path_evaluations.joinpath(cfg.model.name, dataset))
        )


def build_agent(cfg: DictConfig, load_weights_from: Optional[Path] = None) -> BaselineAgent:
    """
    Given a model path, builds an agent with a topology policy that can be used to search for promising actions.

    :param cfg: the hydra config
    :param load_weights_from: the path to the model to load
    :return: a baseline agent using the trained topology policy
    """
    dqn = model_setup(cfg, load_weights_from)
    return BaselineAgent(
        grid2op.make(cfg.env.env_name).action_space,
        TopoPolicyStableDQN(dqn),
    )


def model_setup(cfg: DictConfig, load_weights_from: Optional[Path] = None) -> DQN:
    # create grid2opWrapperEnvironment from hydra config using action and observation spaces from the config
    env: Grid2OpEnvWrapper = instantiate(
        cfg.env,
        obs_space_creation = lambda e: instantiate(cfg.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation = lambda e: instantiate(cfg.act_space, grid2op_action_space=e.action_space)
    )

    # model
    hacky_feature_extractor_kwargs = {}
    # the class is stored as str and after too much time trying to do better I decided to keep it like this
    if cfg.model.sb3.policy_kwargs.get("features_extractor_class") == "SB3GNNWrapper":
        from common.GNN import SB3GNNWrapper
        hacky_feature_extractor_kwargs["policy_kwargs"] = {
            "features_extractor_class": SB3GNNWrapper
        }
    elif cfg.model.sb3.policy_kwargs.get("features_extractor_class") is not None:
        raise ValueError("Unknown feature extractor class" + cfg.model.sb3.policy_kwargs.get("features_extractor_class"))

    # check if cuda is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dqn: DQN = instantiate(
        cfg.model.sb3,
        env=env,
        device=device,
        tensorboard_log=base_path_logs.joinpath(cfg.model.name),
        **hacky_feature_extractor_kwargs
    )

    # load weights
    if load_weights_from is not None:
        dqn.load(load_weights_from, custom_objects=dict(action_space=env.action_space, observation_space=env.observation_space))
    return dqn



@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    train(cfg)
    evaluate(cfg)


if __name__ == '__main__':
    main()
