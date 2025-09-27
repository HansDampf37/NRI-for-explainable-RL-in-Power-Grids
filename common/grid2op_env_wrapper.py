from typing import Optional

import grid2op
from grid2op.gym_compat import DiscreteActSpace, BoxGymObsSpace
from gymnasium import Env
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from lightsim2grid import LightSimBackend


class Grid2OpEnvWrapper(Env):
    """
    Gymnasium-compatible wrapper for Grid2Op environments.

    This class wraps a Grid2Op environment and exposes it through a standard
    Gymnasium interface. This wrapper implements the same logic as GymEnvWithRecoWithDN
    (automatically reconnect powerlines do nothing if load is low).

    """

    def __init__(self,
                 env_name: str = "l2rpn_case14_sandbox",
                 safe_max_rho: float = 0.95,
                 action_space_creation=lambda env: DiscreteActSpace(env.action_space),
                 observation_space_creation=lambda env: BoxGymObsSpace(grid2op_observation_space=env.observation_space)):
        super().__init__()
        # === Grid2Op environment setup ===
        self._g2op_env = grid2op.make(env_name, backend=LightSimBackend())
        self._gym_env = GymEnvWithRecoWithDN(self._g2op_env, safe_max_rho=safe_max_rho)

        # === Observation space setup ===
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = observation_space_creation(self._g2op_env)

        # === Action space setup ===
        self._gym_env.action_space.close()
        self._gym_env.action_space = action_space_creation(self._g2op_env)

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space
        self.g2op_observation_space = self._g2op_env.observation_space
        self.g2op_action_space = self._g2op_env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self._gym_env.reset(seed=seed, options=options)

    def step(self, action):
        return self._gym_env.step(action)
