import os.path
import uuid
from datetime import datetime
from pathlib import Path

import grid2op
import hydra
from grid2op.gym_compat import BoxGymObsSpace
from hydra.utils import instantiate
from lightsim2grid import LightSimBackend
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN, PPO

from baselines.baseline_agent import BaselineAgent, evaluate_agent
from common import G2OpGymEnv
from common.constants import LOGS_PATH, MODELS_PATH, EVAL_PATH
from common.rewards import MazeRLReward
from nri.agent.dqn.DQNTopoPolicy import Sb3DQNTopologyPolicy


def get_env(cfg) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action space from hydra config. Observation space is always
    BoxGymObsSpace since we will use an MLP instead of a GNN

    :param cfg: The hydra config
    :return: The environment
    """
    env: G2OpGymEnv = instantiate(
        cfg.env.training_env,
        obs_space_creation=lambda e: BoxGymObsSpace(grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.ra_dqn.act_space, grid2op_action_space=e.action_space)
    )
    return env


@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    group = "relation-unaware-baselines/ppo"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    name = f"mlp_{timestamp}_{uuid.uuid4().hex}"
    env = get_env(cfg)

    algorithm = PPO(
        env=env,
        policy="MlpPolicy",
        tensorboard_log=os.path.join(LOGS_PATH, group),
        ent_coef=0.05,
    )
    algorithm.learn(total_timesteps=600000, tb_log_name=name, log_interval=cfg.ra_dqn.train.log_interval)
    algorithm.save(os.path.join(MODELS_PATH, group, name))


if __name__ == "__main__":
    main()
