import json
import logging
import os
from pathlib import Path
from typing import Optional

import grid2op
import hydra
from hydra.utils import instantiate
from lightsim2grid import LightSimBackend
from omegaconf import DictConfig
from stable_baselines3 import DQN

from baselines.baseline_agent import BaselineAgent, evaluate_agent
from baselines.topo_policy_stable_dqn import TopoPolicyStableDQN
from common import Grid2OpEnvWrapper
from common.grid2op_env_wrapper import get_env

logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_path_models: Path = Path("data/models/stable-baselines/")
base_path_logs: Path = Path("data/logs/stable-baselines/")
base_path_evaluations: Path = Path("data/evaluations/stable-baselines/")


def train(cfg: DictConfig):
    """
    Creates a model using the setup_method, trains it and saves it.
    """
    dqn = model_setup(cfg)
    try:
        dqn.learn(cfg.baseline.train.timesteps, log_interval=10)
    except KeyboardInterrupt:
        answered = False
        while not answered:
            save = input("You interrupted the training. Do you want to safe the model? (Y/N)")
            if save.lower() in ["n", "no"]:
                raise KeyboardInterrupt()
            elif save.lower() in ["y", "yes"]:
                dqn.save(Path(base_path_models.joinpath(cfg.baseline.model.name)))
                raise KeyboardInterrupt()

    dqn.save(Path(base_path_models.joinpath(cfg.baseline.model.name)))


def evaluate(cfg: DictConfig):
    """
    Loads and evaluates a trained topology policy. Results are stored under data/evaluation.
    :param cfg: The hydra configuration.
    """
    dqn: DQN = model_setup(cfg, load_weights_from=Path(base_path_models.joinpath(cfg.baseline.model.name)))
    env: Grid2OpEnvWrapper = get_env(cfg)

    for _ in range(cfg.baseline.eval.nb_episodes):
        obs, info = env.reset()
        cumulative_reward = 0
        episode_length = 0
        while True:
            act, _ = dqn.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(act.item())
            if done or truncated:
                break
            cumulative_reward += reward
            episode_length += 1
        print(f"Survived {episode_length} steps with a return of {cumulative_reward:.2f}")
        name_chronic = os.path.basename(info['time_series_id'])
        base_path = Path(base_path_evaluations.joinpath(cfg.baseline.model.name + "_topology_policy", name_chronic))
        base_path.mkdir(parents=True, exist_ok=True)
        with open(base_path.joinpath("episode_meta.json"), 'w') as f:
            json.dump({
            "agent_seed": None,
            "chronics_max_timestep": -1,
            "cumulative_reward": cumulative_reward,
            "nb_timestep_played": episode_length
        }, f, indent=4)

    for dataset in ["train", "test", "val"]:
        env = grid2op.make(f"{cfg.env.env_name}_{dataset}", backend=LightSimBackend())
        evaluate_agent(
            agent=build_agent(cfg, Path(base_path_models.joinpath(cfg.baseline.model.name))),
            env=env,
            num_episodes=cfg.baseline.eval.nb_episodes, # len(env.chronics_handler.subpaths),  # run all episodes
            path_results=Path(base_path_evaluations.joinpath(cfg.baseline.model.name + "_agent", dataset))
        )


def build_agent(cfg: DictConfig, load_weights_from: Optional[Path] = None) -> BaselineAgent:
    """
    Given a hydra config and a place to load the model weights from returns a reproducible Agent.

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
    """
    Given the hydra config and a place to load the model weights from returns a reproducible DQN.

    :param cfg: the hydra config
    :param load_weights_from: the path to the model to load
    """
    # create grid2opWrapperEnvironment from hydra config using action and observation spaces from the config
    env: Grid2OpEnvWrapper = get_env(cfg)

    # model
    hacky_feature_extractor_kwargs = {}
    # the class is stored as str and after too much time trying to do better I decided to keep it like this
    if cfg.baseline.model.sb3.policy_kwargs.get("features_extractor_class") == "SB3GNNWrapper":
        from common.GNN import SB3GNNWrapper
        hacky_feature_extractor_kwargs["policy_kwargs"] = {
            "features_extractor_class": SB3GNNWrapper
        }
    elif cfg.baseline.model.sb3.policy_kwargs.get("features_extractor_class") is not None:
        raise ValueError("Unknown feature extractor class" + cfg.baseline.model.sb3.policy_kwargs.get("features_extractor_class"))

    dqn: DQN = instantiate(
        cfg.baseline.model.sb3,
        env=env,
        tensorboard_log=base_path_logs.joinpath(cfg.baseline.model.name),
        **hacky_feature_extractor_kwargs
    )

    # load weights
    if load_weights_from is not None:
        dqn.set_parameters(load_weights_from)
    return dqn


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    train(cfg)
    evaluate(cfg)


if __name__ == '__main__':
    main()
