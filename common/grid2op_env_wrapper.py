from typing import Optional

import grid2op
from grid2op.gym_compat import DiscreteActSpace, BoxGymObsSpace
from gymnasium import Env
from hydra.utils import instantiate
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from lightsim2grid import LightSimBackend

from common.rewards import MazeRLReward


class Grid2OpEnvWrapper(Env):
    """
    Gymnasium-compatible wrapper for Grid2Op environments with heuristic actions

    This class wraps a Grid2Op environment and exposes it through a standard
    Gymnasium interface. This wrapper implements the same logic as GymEnvWithRecoWithDN
    (automatically reconnect powerlines do nothing if load is low).
    """

    def __init__(self,
                 env_name: str = "l2rpn_case14_sandbox",
                 safe_max_rho: float = 0.95,
                 act_space_creation=lambda env: DiscreteActSpace(env.action_space),
                 obs_space_creation=lambda env: BoxGymObsSpace(grid2op_observation_space=env.observation_space)):
        super().__init__()
        self._g2op_env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=MazeRLReward)
        self._gym_env = GymEnvWithRecoWithDN(self._g2op_env, safe_max_rho=safe_max_rho)

        self._gym_env.observation_space.close()
        self._gym_env.observation_space = obs_space_creation(self._g2op_env)

        self._gym_env.action_space.close()
        self._gym_env.action_space = act_space_creation(self._g2op_env)

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space
        self.g2op_observation_space = self._g2op_env.observation_space
        self.g2op_action_space = self._g2op_env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self._gym_env.reset(seed=seed, options=options)

    def step(self, action):
        return self._gym_env.step(action)


def get_env(cfg):
    """
    Creates a Grid2opWrapperEnvironment from hydra config using action and observation spaces from the config

    :param cfg: The hydra config
    :return: The environment
    """
    env: Grid2OpEnvWrapper = instantiate(
        cfg.env,
        obs_space_creation=lambda e: instantiate(cfg.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.act_space, grid2op_action_space=e.action_space)
    )
    return env
