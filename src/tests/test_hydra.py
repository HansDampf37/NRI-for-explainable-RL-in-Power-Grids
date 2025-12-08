import unittest

import grid2op
import hydra
from grid2op.gym_compat import DiscreteActSpace
from hydra.utils import instantiate

from src.common.action_space import ReducedActionSpace
from src.common.env import G2OpGymEnv
from src.common.graph_structured_observation_space import BusConnectivityGraphObsSpace


class TestHydra(unittest.TestCase):
    def test_hydra_obs_space(self):
        with hydra.initialize(config_path="../../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(config_name="rl/obs_space/bus_connectivity_graph_obs_space")
            env = grid2op.make("l2rpn_case14_sandbox")
            obs_space = instantiate(cfg.rl.obs_space, grid2op_observation_space=env.observation_space)
            self.assertIsInstance(obs_space, BusConnectivityGraphObsSpace)

    def test_hydra_act_space(self):
        with hydra.initialize(config_path="../../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(config_name="rl/act_space/discrete_set_bus")
            env = grid2op.make("l2rpn_case14_sandbox")
            act_space = instantiate(cfg.rl.act_space, grid2op_action_space=env.action_space)
            self.assertIsInstance(act_space, DiscreteActSpace)

    def test_hydra_reduced_act_space(self):
        with hydra.initialize(config_path="../../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(config_name="rl/act_space/reduced_act_space")
            env = grid2op.make("l2rpn_case14_sandbox")
            act_space = instantiate(cfg.rl.act_space, grid2op_action_space=env.action_space)
            self.assertIsInstance(act_space, ReducedActionSpace)

    def test_hydra_env_creation(self):
        with hydra.initialize(config_path="../../hydra_configs", version_base="1.3"):
            env_cfg = hydra.compose(config_name="env/case14")
            obs_cfg = hydra.compose(config_name="rl/obs_space/bus_connectivity_graph_obs_space")
            act_cfg = hydra.compose(config_name="rl/act_space/reduced_act_space")
            env = instantiate(
                env_cfg.env.training_env,
                obs_space_creation = lambda e: instantiate(obs_cfg.rl.obs_space, grid2op_observation_space=e.observation_space),
                act_space_creation = lambda e: instantiate(act_cfg.rl.act_space, grid2op_action_space=e.action_space)
            )
            self.assertIsInstance(env, G2OpGymEnv)
