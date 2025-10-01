import unittest

import grid2op
import hydra
from grid2op.gym_compat import DiscreteActSpace, BoxGymObsSpace
from hydra.utils import instantiate

from common import Grid2OpEnvWrapper
from common.graph_structured_observation_space import GraphObservationSpace

class TestHydra(unittest.TestCase):
    def test_hydra_obs_space(self):
        with hydra.initialize(config_path="../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(config_name="baseline/obs_space/graph_obs_space")
            env = grid2op.make("l2rpn_case14_sandbox")
            obs_space = instantiate(cfg.baseline.obs_space, grid2op_observation_space=env.observation_space)
            self.assertIsInstance(obs_space, GraphObservationSpace)

    def test_hydra_act_space(self):
        with hydra.initialize(config_path="../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(config_name="baseline/act_space/discrete_set_bus")
            env = grid2op.make("l2rpn_case14_sandbox")
            act_space = instantiate(cfg.baseline.act_space, grid2op_action_space=env.action_space)
            self.assertIsInstance(act_space, DiscreteActSpace)

    def test_hydra_obs_space_2(self):
        with hydra.initialize(config_path="../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(config_name="baseline/obs_space/box_obs_space")
            env = grid2op.make("l2rpn_case14_sandbox")
            obs_space = instantiate(cfg.baseline.obs_space, grid2op_observation_space=env.observation_space)
            self.assertIsInstance(obs_space, BoxGymObsSpace)

    def test_hydra_env_creation(self):
        with hydra.initialize(config_path="../hydra_configs", version_base="1.3"):
            env_cfg = hydra.compose(config_name="env/case14")
            obs_cfg = hydra.compose(config_name="baseline/obs_space/box_obs_space")
            act_cfg = hydra.compose(config_name="baseline/act_space/discrete_set_bus")
            env = instantiate(
                env_cfg.env,
                obs_space_creation = lambda e: instantiate(obs_cfg.baseline.obs_space, grid2op_observation_space=e.observation_space),
                act_space_creation = lambda e: instantiate(act_cfg.baseline.act_space, grid2op_action_space=e.action_space)
            )
            self.assertIsInstance(env, Grid2OpEnvWrapper)
