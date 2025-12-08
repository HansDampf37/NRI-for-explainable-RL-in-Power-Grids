import unittest
from typing import Dict

from grid2op.gym_compat import BoxGymObsSpace, BoxGymActSpace
from gymnasium.spaces import Discrete, Box

from src.common.graph_structured_observation_space import BusConnectivityGraphObsSpace
from src.common.env import G2OpGymEnv


class TestGrid2opEnvWrapper(unittest.TestCase):
    def setUp(self):
        self.env = G2OpGymEnv()
        self.env.reset()

    def test_custom_obs_space(self):
        env = G2OpGymEnv(obs_space_creation=lambda e: BusConnectivityGraphObsSpace(e.observation_space))
        obs, info = env.reset()
        self.assertIsInstance(obs, Dict)

    def test_config_1(self):
        self.assertIsInstance(self.env.action_space, Discrete)
        self.assertIsInstance(self.env.observation_space, Box)
        self.assertEqual(self.env._g2op_env.n_gen, 6)  # small env by default

    def test_config_2(self):
        self.env = G2OpGymEnv(
            env_name="l2rpn_idf_2023",
            act_space_creation=lambda e: BoxGymActSpace(e.action_space),
            obs_space_creation=lambda e: BoxGymObsSpace(e.observation_space, attr_to_keep=["rho"]))
        self.assertEqual(self.env._g2op_env.n_gen, 62)
        self.assertIsInstance(self.env.action_space, Box)
        self.assertIsInstance(self.env.observation_space, Box)
        self.assertEqual(self.env._gym_env.observation_space._attr_to_keep, ["rho"])

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_step(self):
        obs, rew, done, truncated, info = self.env.step(self.env.action_space.sample())
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIsInstance(rew, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)

    def test_to_nothing(self):
        self.env.do_nothing() # test that it works in the beginning of the episodes it should not fail
