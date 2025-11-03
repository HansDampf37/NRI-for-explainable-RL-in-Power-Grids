"""
This script wraps a grid2op environment with a gymnasium API and applies heuristic actions automatically.
"""
from typing import Optional

import grid2op
from grid2op.gym_compat import DiscreteActSpace, BoxGymObsSpace
from gymnasium import Env
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from lightsim2grid import LightSimBackend

from common.rewards import MazeRLReward


class G2OpGymEnv(Env):
    """
    Gymnasium-compatible wrapper for Grid2Op environments with heuristic actions.

    This class wraps a Grid2Op environment and exposes it through a standard Gymnasium interface.
    This wrapper implements the same logic as GymEnvWithRecoWithDN (automatically reconnect powerlines do nothing if load is low).
    Additionally, the do-nothing action is applied
    whenever the maximum line load is lower than safe_max_rho.
    """

    def __init__(self,
                 env_name: str = "l2rpn_case14_sandbox",
                 safe_max_rho: float = 0.95,
                 act_space_creation=lambda env: DiscreteActSpace(env.action_space, attr_to_keep=["set_bus"]),
                 obs_space_creation=lambda env: BoxGymObsSpace(grid2op_observation_space=env.observation_space)):
        """
        Constructor.
        @param env_name: the name of the grid2op environment
        @param safe_max_rho: do nothing if max rho is below this value
        @param act_space_creation: lambda function that creates the action space
        @param obs_space_creation: lambda function that creates the observation space
        """
        super().__init__()
        # create env
        self._g2op_env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=MazeRLReward)
        self._gym_env = GymEnvWithRecoWithDN(self._g2op_env, safe_max_rho=safe_max_rho, with_forecast=True)

        # create observation space
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = obs_space_creation(self._g2op_env)
        self.observation_space = self._gym_env.observation_space
        self.g2op_observation_space = self._g2op_env.observation_space

        # create action space
        self._gym_env.action_space.close()
        self._gym_env.action_space = act_space_creation(self._g2op_env)
        self.action_space = self._gym_env.action_space
        self.g2op_action_space = self._g2op_env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self._gym_env.reset(seed=seed, options=options)

    def step(self, action):
        return self._gym_env.step(action)
