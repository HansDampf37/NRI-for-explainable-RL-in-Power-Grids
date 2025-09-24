import copy
from typing import Dict, Literal, Any, Optional

import grid2op
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from lightsim2grid import LightSimBackend


class Grid2OpEnvWrapper(Env):
    """
    Gymnasium-compatible wrapper for Grid2Op environments.

    This class wraps a Grid2Op environment and exposes it through a standard
    Gymnasium interface, including support for custom observation and action
    spaces (discrete, box, and multi-discrete). Furthermore, this wrapper is compatible
    with RLLib API. This wrapper implements the same logic as GymEnvWithRecoWithDN
    (automatically reconnect powerlines do nothing if load is low).

    :param env_config: Optional configuration dictionary with the following keys:
        - "backend_cls" (type): Backend class to use (default: LightSimBackend).
        - "backend_options" (dict): Options passed to the backend constructor.
        - "env_name" (str): Name of the Grid2Op environment (default: "l2rpn_case14_sandbox").
        - "env_is_test" (bool): Whether to use the test environment (default: False).
        - "obs_attr_to_keep" (list): List of observation attributes to keep.
        - "act_type" (str): Action space type: "discrete", "box", or "multi_discrete".
        - "act_attr_to_keep" (list): List of action attributes to keep.
    :type env_config: Optional[Dict[str, Any]]
    """

    def __init__(self,
                 env_config: Optional[Dict[Literal[
                     "backend_cls",
                     "backend_options",
                     "env_name",
                     "env_is_test",
                     "obs_attr_to_keep",
                     "act_type",
                     "act_attr_to_keep",
                     "safe_max_rho"], Any]] = None):
        super().__init__()
        env_config = env_config or {}

        # === Backend setup ===
        backend_cls = env_config.get("backend_cls", LightSimBackend)
        backend_options = env_config.get("backend_options", {})
        backend = backend_cls(**backend_options)

        # === Grid2Op environment setup ===
        env_name = env_config.get("env_name", "l2rpn_case14_sandbox")
        is_test = bool(env_config.get("env_is_test", False))
        safe_max_rho = env_config.get("safe_max_rho", 0.95)
        self._g2op_env = grid2op.make(env_name, backend=backend, test=is_test)
        self._gym_env = GymEnvWithRecoWithDN(self._g2op_env, safe_max_rho=safe_max_rho)

        # === Observation space setup ===
        obs_attr_to_keep = copy.deepcopy(env_config.get("obs_attr_to_keep", ["rho", "p_or", "gen_p", "load_p"]))
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(
            self._g2op_env.observation_space,
            attr_to_keep=obs_attr_to_keep
        )
        self.observation_space = Box(
            shape=self._gym_env.observation_space.shape,
            low=self._gym_env.observation_space.low,
            high=self._gym_env.observation_space.high
        )

        # === Action space setup ===
        act_type = env_config.get("act_type", "discrete")
        act_attr_to_keep = copy.deepcopy(env_config.get("act_attr_to_keep", []))

        self._gym_env.action_space.close()

        if act_type == "discrete":
            if not act_attr_to_keep:
                act_attr_to_keep = ["set_line_status_simple", "set_bus"]
            self._gym_env.action_space = DiscreteActSpace(
                self._g2op_env.action_space,
                attr_to_keep=act_attr_to_keep
            )
            self.action_space = Discrete(self._gym_env.action_space.n)

        elif act_type == "box":
            if not act_attr_to_keep:
                act_attr_to_keep = ["redispatch", "set_storage", "curtail"]
            self._gym_env.action_space = BoxGymActSpace(
                self._g2op_env.action_space,
                attr_to_keep=act_attr_to_keep
            )
            self.action_space = Box(
                shape=self._gym_env.action_space.shape,
                low=self._gym_env.action_space.low,
                high=self._gym_env.action_space.high
            )

        elif act_type == "multi_discrete":
            if not act_attr_to_keep:
                act_attr_to_keep = ["one_line_set", "one_sub_set"]
            self._gym_env.action_space = MultiDiscreteActSpace(
                self._g2op_env.action_space,
                attr_to_keep=act_attr_to_keep
            )
            self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)

        else:
            raise NotImplementedError(f"Action type '{act_type}' is not supported.")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self._gym_env.reset(seed=seed, options=options)

    def step(self, action):
        return self._gym_env.step(action)
