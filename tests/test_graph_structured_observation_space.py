import unittest

import grid2op
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from stable_baselines3 import DQN

from common.graph_structured_observation_space import GraphObservationSpace, EDGE_INDEX, \
    EDGES, NODES, BipartitGraphObservationSpace


class TestGraphStructuredObservationSpace(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.gym_env = GymEnv(self.env)
        self.gym_env.observation_space.close()
        self.gym_env.observation_space = GraphObservationSpace(self.env.observation_space)
        self.obs_space = self.gym_env.observation_space

    def test_observation_space(self):
        obs, _ = self.gym_env.reset()
        self.assertIn(NODES, obs.keys())
        self.assertIn(EDGES, obs.keys())
        self.assertIn(EDGE_INDEX, obs.keys())

    def test_stable_baselines_compatibility(self):
        self.gym_env.action_space.close()
        self.gym_env.observation_space = GraphObservationSpace(self.env.observation_space,
                                                               spaces_to_keep=[NODES])
        self.gym_env.action_space = DiscreteActSpace(self.env.action_space, attr_to_keep=["set_bus"])
        dqn = DQN("MultiInputPolicy", env=self.gym_env)
        dqn.learn(total_timesteps=100)

    def test_line_features_from_obs(self):
        obs = self.env.reset()
        line_features = self.obs_space.line_features_from_observation(obs)
        self.assertEqual(line_features.shape, (self.env.n_line, self.obs_space.NUM_FEATURES_PER_LINE))

    def test_node_features_from_obs(self):
        obs = self.env.reset()
        num_nodes = self.env.n_gen + self.env.n_load + self.env.n_sub
        node_features = self.obs_space.node_features_from_observation(obs)
        self.assertEqual(node_features.shape, (num_nodes, self.obs_space.NUM_FEATURES_PER_NODE))

    def test_edge_features_from_obs(self):
        obs = self.env.reset()
        num_edges = self.env.n_line + self.env.n_gen + self.env.n_load
        edge_features = self.obs_space.edge_features_from_observation(obs)
        self.assertEqual(edge_features.shape, (num_edges, self.obs_space.NUM_FEATURES_PER_EDGE))


class TestBipartitGraphStructuredObservationSpace(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.gym_env = GymEnv(self.env)
        self.gym_env.observation_space.close()
        self.gym_env.observation_space = BipartitGraphObservationSpace(self.env.observation_space)
        self.obs_space = self.gym_env.observation_space

    def test_observation_space(self):
        obs, _ = self.gym_env.reset()
        self.assertIn(NODES, obs.keys())
        self.assertIn(EDGE_INDEX, obs.keys())
        self.assertNotIn(EDGES, obs.keys())

    def test_stable_baselines_compatibility(self):
        self.gym_env.action_space.close()
        self.gym_env.observation_space = BipartitGraphObservationSpace(self.env.observation_space)
        self.gym_env.action_space = DiscreteActSpace(self.env.action_space, attr_to_keep=["set_bus"])
        dqn = DQN("MultiInputPolicy", env=self.gym_env)
        dqn.learn(total_timesteps=100)

    def test_obs_shape(self):
        obs = self.env.reset()
        d = self.obs_space.to_gym(obs)
        target_feature_dim = self.obs_space.graph_obs_space.NUM_FEATURES_PER_NODE + self.obs_space.graph_obs_space.NUM_FEATURES_PER_EDGE
        self.assertEqual(d[NODES].shape, (self.obs_space.n_node_bipart, target_feature_dim))
        self.assertEqual(d[EDGE_INDEX].shape, (2, self.obs_space.n_edge_bipart))
