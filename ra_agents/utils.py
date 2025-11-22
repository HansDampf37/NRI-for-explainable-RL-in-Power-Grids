from typing import Optional

from grid2op.gym_compat import BoxGymObsSpace
from hydra.utils import instantiate

from common.env import G2OpGymEnv


def get_env(cfg, env_name: Optional[str] = None) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action and observation spaces from hydra config.

    :param cfg: The hydra config
    :param env_name: Optional override for the environment name
    :return: The environment
    """
    env: G2OpGymEnv = G2OpGymEnv(
        cfg.env.training_env.env_name if env_name is None else env_name,
        obs_space_creation=lambda e: instantiate(cfg.rl.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space)
    )
    return env

def get_env_mlp_baseline(cfg, env_name: Optional[str] = None) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action and observation spaces from hydra config.

    :param cfg: The hydra config
    :param env_name: Optional override for the environment name
    :return: The environment
    """
    env: G2OpGymEnv = G2OpGymEnv(
        cfg.env.training_env.env_name if env_name is None else env_name,
        obs_space_creation=lambda e: BoxGymObsSpace(grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space)
    )
    return env