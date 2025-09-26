import unittest

import grid2op
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from stable_baselines3 import DQN

from common.graph_structured_observation_space import GraphStructuredBoxObservationSpace, FEATURES_PER_LINE, \
    FEATURES_PER_LOAD, FEATURES_PER_GENERATOR, FEATURES_PER_NODE


class TestGraphStructuredObservationSpace(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.gym_env = GymEnv(self.env)
        self.gym_env.observation_space.close()
        self.gym_env.observation_space = GraphStructuredBoxObservationSpace()
        self.obs_space = self.gym_env.observation_space

    def test_observation_space(self):
        obs, _ = self.gym_env.reset()
        self.assertIn("node_features", obs.keys())
        self.assertIn("edge_features", obs.keys())
        self.assertIn("edge_index", obs.keys())

    def test_stable_baselines_compatibility(self):
        self.gym_env.action_space.close()
        self.gym_env.action_space = DiscreteActSpace(self.env.action_space, attr_to_keep=["set_bus"])
        dqn = DQN("MultiInputPolicy", env=self.gym_env)
        dqn.learn(total_timesteps=100)

    def test_line_features_from_obs(self):
        obs = self.env.reset()
        line_features = self.obs_space.line_features_from_observation(obs)
        self.assertEqual(line_features.shape, (self.env.n_line, FEATURES_PER_LINE))

    def test_load_features_from_obs(self):
        obs = self.env.reset()
        load_features = self.obs_space.load_features_from_observation(obs)
        self.assertEqual(load_features.shape, (self.env.n_load, FEATURES_PER_LOAD))

    def test_generator_features_from_obs(self):
        obs = self.env.reset()
        gen_features = self.obs_space.generator_features_from_observation(obs)
        self.assertEqual(gen_features.shape, (self.env.n_gen, FEATURES_PER_GENERATOR))

    def test_node_features_from_obs(self):
        observation, _ = self.gym_env.reset()
        self.assertEqual(observation["node_features"].shape,
                         (self.env.n_gen + self.env.n_load + self.env.n_sub, FEATURES_PER_NODE))

    def test_edge_features_from_obs(self):
        observation, _ = self.gym_env.reset()
        self.assertEqual(observation["edge_features"].shape, (self.env.n_line, FEATURES_PER_LINE))
