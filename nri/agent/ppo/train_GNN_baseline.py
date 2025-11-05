import os
import uuid
from datetime import datetime
from pathlib import Path

import grid2op
import hydra
from hydra.utils import instantiate
from lightsim2grid import LightSimBackend
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN

from baselines import evaluate_agent
from baselines.baseline_agent import BaselineAgent
from common import G2OpGymEnv
from common.constants import LOGS_PATH, EVAL_PATH
from common.rewards import MazeRLReward
from nri.agent.RAFeatureExtractor import BaselineFeatureExtractorSB3
from nri.agent.dqn.DQNTopoPolicy import Sb3DQNTopologyPolicy


def get_env(cfg) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action and observation spaces from hydra config

    :param cfg: The hydra config
    :return: The environment
    """
    env: G2OpGymEnv = instantiate(
        cfg.env.training_env,
        obs_space_creation=lambda e: instantiate(cfg.ra_dqn.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.ra_dqn.act_space, grid2op_action_space=e.action_space)
    )
    return env


@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    group = "relation-unaware-baselines/ppo"
    name = f"gnn_{timestamp}_{uuid.uuid4().hex}"
    env = get_env(cfg)

    policy_kwargs = {
        "net_arch": cfg.ra_dqn.model.sb3.policy_kwargs.net_arch,
        "features_extractor_class": BaselineFeatureExtractorSB3,
        "features_extractor_kwargs": {
            "hidden_dim": cfg.ra_dqn.model.sb3.policy_kwargs.features_extractor_kwargs.hidden_dim,
            "out_dim": cfg.ra_dqn.model.sb3.policy_kwargs.features_extractor_kwargs.out_dim,
            "dropout_prob": cfg.ra_dqn.model.sb3.policy_kwargs.features_extractor_kwargs.dropout_prob,
            "num_layers": cfg.ra_dqn.model.sb3.policy_kwargs.features_extractor_kwargs.num_layers,
        }
    }

    # train
    algorithm = DQN(
        env=env,
        policy="MultiInputPolicy",
        tensorboard_log=os.path.join(LOGS_PATH, group),
        policy_kwargs=policy_kwargs,
        verbose=cfg.ra_dqn.model.sb3.verbose,
        train_freq=cfg.ra_dqn.model.sb3.train_freq,
        gradient_steps=cfg.ra_dqn.model.sb3.gradient_steps,
        gamma=cfg.ra_dqn.model.sb3.gamma,
        exploration_fraction=cfg.ra_dqn.model.sb3.exploration_fraction,
        exploration_final_eps=cfg.ra_dqn.model.sb3.exploration_final_eps,
        target_update_interval=cfg.ra_dqn.model.sb3.target_update_interval,
        learning_starts=cfg.ra_dqn.model.sb3.learning_starts,
        buffer_size=cfg.ra_dqn.model.sb3.buffer_size,
        batch_size=cfg.ra_dqn.model.sb3.batch_size,
        learning_rate=cfg.ra_dqn.model.sb3.learning_rate,
    )
    algorithm.learn(total_timesteps=cfg.ra_dqn.train.timesteps, tb_log_name=name, log_interval=cfg.ra_dqn.train.log_interval)
    algorithm.save(os.path.join(LOGS_PATH, group, name))

    # evaluate
    agent = BaselineAgent(
        env.g2op_action_space,
        Sb3DQNTopologyPolicy(algorithm)
    )
    for dataset in ["train", "test", "val"]:
        grid2op_env = grid2op.make(f"{cfg.env.env_name}_{dataset}", backend=LightSimBackend(), reward_class=MazeRLReward)
        evaluate_agent(
            agent=agent,
            env=grid2op_env,
            num_episodes=cfg.baseline.eval.nb_episodes,
            path_results=Path(os.path.join(EVAL_PATH, group, name + "_" + dataset))
        )


if __name__ == "__main__":
    main()
