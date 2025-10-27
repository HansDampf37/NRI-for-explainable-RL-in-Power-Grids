import unittest

import grid2op
import numpy as np
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from stable_baselines3 import DQN
from torch_geometric.data.data import Data

from common.graph_structured_observation_space import EntityGraphObservationSpace, EDGE_INDEX, \
    EDGES, NODES, BipartitGraphObservationSpace, EDGE_MASK, BusConnectivityGraphObsSpace, gym2pytorch_geometric_data


class TestGraphStructuredObservationSpace(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.gym_env = GymEnv(self.env)
        self.gym_env.observation_space = EntityGraphObservationSpace(self.env.observation_space)
        self.obs_space = self.gym_env.observation_space

    def test_observation_space(self):
        obs, _ = self.gym_env.reset()
        self.assertSetEqual({NODES, EDGES, EDGE_INDEX, EDGE_MASK}, set(obs.keys()))
        self.assertEqual(obs[NODES].shape, (self.obs_space.num_nodes, self.obs_space.x_dim))
        self.assertEqual(obs[EDGES].shape, (self.obs_space.max_num_edges, self.obs_space.e_dim))
        self.assertEqual(obs[EDGE_INDEX].shape, (2, self.obs_space.max_num_edges))
        self.assertEqual(obs[EDGE_MASK].shape, (self.obs_space.max_num_edges, ))

    def test_stable_baselines_compatibility(self):
        self.gym_env.action_space.close()
        self.gym_env.observation_space = EntityGraphObservationSpace(self.env.observation_space, spaces_to_keep=[NODES])
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
        self.assertEqual(node_features.shape, (num_nodes, self.obs_space.x_dim))

    def test_edge_features_from_obs(self):
        obs = self.env.reset()
        num_edges = self.env.n_line + self.env.n_gen + self.env.n_load
        edge_features = self.obs_space.edge_features_from_observation(obs)
        self.assertEqual(edge_features.shape, (num_edges, self.obs_space.e_dim))


class TestBipartitGraphStructuredObservationSpace(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.gym_env = GymEnv(self.env)
        self.gym_env.observation_space = BipartitGraphObservationSpace(self.env.observation_space)
        self.obs_space: BipartitGraphObservationSpace = self.gym_env.observation_space

    def test_observation_space(self):
        obs, _ = self.gym_env.reset()
        self.assertSetEqual({NODES, EDGE_INDEX, EDGE_MASK}, set(obs.keys()))
        self.assertEqual(obs[NODES].shape, (self.obs_space.num_nodes, self.obs_space.x_dim))
        self.assertEqual(obs[EDGE_INDEX].shape, (2, self.obs_space.max_num_edges))
        self.assertEqual(obs[EDGE_MASK].shape, (self.obs_space.max_num_edges,))

    def test_stable_baselines_compatibility(self):
        self.gym_env.action_space.close()
        self.gym_env.observation_space = BipartitGraphObservationSpace(self.env.observation_space)
        self.gym_env.action_space = DiscreteActSpace(self.env.action_space, attr_to_keep=["set_bus"])
        dqn = DQN("MultiInputPolicy", env=self.gym_env)
        dqn.learn(total_timesteps=100)

    def test_obs_shape(self):
        obs = self.env.reset()
        gym_obs = self.obs_space.to_gym(obs)
        target_feature_dim = self.obs_space.graph_obs_space.x_dim + self.obs_space.graph_obs_space.e_dim
        self.assertEqual(gym_obs[NODES].shape, (self.obs_space.num_nodes, target_feature_dim))
        self.assertEqual(gym_obs[EDGE_INDEX].shape, (2, self.obs_space.max_num_edges))

class TestBusConnectivityGraphObsSpace(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.gym_env = GymEnv(self.env, with_forecast=True)
        self.gym_env.observation_space = BusConnectivityGraphObsSpace(self.env.observation_space)
        self.obs_space = self.gym_env.observation_space

    def test_observation_space(self):
        obs, _ = self.gym_env.reset()
        self.assertSetEqual({NODES, EDGE_INDEX, EDGE_MASK}, set(obs.keys()))
        self.assertEqual(obs[NODES].shape, (self.obs_space.num_nodes, self.obs_space.x_dim))
        self.assertEqual(obs[EDGE_INDEX].shape, (2, self.obs_space.max_num_edges))
        self.assertEqual(obs[EDGE_MASK].shape, (self.obs_space.max_num_edges,))

    def test_obs_shape(self):
        obs, _ = self.gym_env.reset()
        self.assertEqual(obs[NODES].shape, (self.obs_space.num_nodes, self.obs_space.x_dim))
        self.assertEqual(obs[EDGE_INDEX].shape, (2, self.obs_space.max_num_edges))
        self.assertEqual(obs[EDGE_MASK].shape, (self.obs_space.max_num_edges, ))

    def test_data(self):
        obs, _ = self.gym_env.reset()
        data = gym2pytorch_geometric_data(obs)
        self.assertIsInstance(data, Data)
        self.assertLessEqual(data.edge_index.shape[1], self.obs_space.max_num_edges)
        self.assertEqual(data.edge_index.shape[0], 2)
        print(data)

    def test_edge_mask(self):
        obs, _ = self.gym_env.reset()
        true_edges = np.unique(obs[EDGE_MASK], return_counts=True)[1]
        # all edges should be present -> only one edge label with all counts
        self.assertEqual(self.obs_space.max_num_edges, true_edges[0])

    def test_with_and_without_forecasts(self):
        obs_space_with_forecasts = BusConnectivityGraphObsSpace(self.env.observation_space, with_forecast=True)
        obs_space_without_forecasts = BusConnectivityGraphObsSpace(self.env.observation_space, with_forecast=False)
        self.assertIn("active_power_forecast", obs_space_with_forecasts.node_feature_names)
        self.assertNotIn("active_power_forecast", obs_space_without_forecasts.node_feature_names)
        self.assertEqual(obs_space_without_forecasts.x_dim, 6)
        self.assertEqual(obs_space_with_forecasts.x_dim, 8)
        self.assertEqual(obs_space_without_forecasts.to_gym(self.env.reset())[NODES].shape[-1], obs_space_without_forecasts.x_dim)
        self.assertEqual(obs_space_with_forecasts.to_gym(self.env.reset())[NODES].shape[-1], obs_space_with_forecasts.x_dim)
