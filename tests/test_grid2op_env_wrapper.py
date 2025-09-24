import unittest

from gymnasium.spaces import Discrete, Box

from common.grid2op_env_wrapper import Grid2OpEnvWrapper


class TestGrid2opEnvWrapper(unittest.TestCase):
    def setUp(self):
        self.env = Grid2OpEnvWrapper()
        self.env.reset()

    def test_config_1(self):
        self.assertIsInstance(self.env.action_space, Discrete)
        self.assertIsInstance(self.env.observation_space, Box)
        self.assertEqual(self.env._g2op_env.n_gen, 6)  # small env by default

    def test_config_2(self):
        self.env = Grid2OpEnvWrapper(dict(env_name="l2rpn_idf_2023", act_type="box", obs_attr_to_keep=["rho"]))
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
