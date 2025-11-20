import unittest
from collections import Counter

import grid2op
import numpy as np
from grid2op.gym_compat import GymEnv
from torch_geometric.data.data import Data

from common.graph_structured_observation_space import EDGE_INDEX, NODES, EDGE_MASK, BusConnectivityGraphObsSpace, \
    gym2pytorch_geometric_data, GLOBAL


class TestBusConnectivityGraphObsSpace(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.gym_env = GymEnv(self.env, with_forecast=True)
        self.gym_env.observation_space = BusConnectivityGraphObsSpace(self.env.observation_space)
        self.obs_space = self.gym_env.observation_space

    def test_observation_space(self):
        obs, _ = self.gym_env.reset()
        self.assertSetEqual({NODES, EDGE_INDEX, EDGE_MASK, GLOBAL}, set(obs.keys()))
        self.assertEqual(obs[NODES].shape, (self.obs_space.num_nodes, self.obs_space.x_dim))
        self.assertEqual(obs[EDGE_INDEX].shape, (2, self.obs_space.max_num_edges))
        self.assertEqual(obs[EDGE_MASK].shape, (self.obs_space.max_num_edges,))

    def test_obs_shape(self):
        obs, _ = self.gym_env.reset()
        self.assertEqual(obs[NODES].shape, (self.obs_space.num_nodes, self.obs_space.x_dim))
        self.assertEqual(obs[EDGE_INDEX].shape, (2, self.obs_space.max_num_edges))
        self.assertEqual(obs[EDGE_MASK].shape, (self.obs_space.max_num_edges,))

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

    def test_undirected(self):
        obs, _ = self.gym_env.reset()
        cols = list(map(tuple, obs[EDGE_INDEX].T))
        c = Counter(cols)
        missing_mask = np.array([c[(y, x)] == 0 for (x, y) in cols])
        self.assertTrue(np.all(np.logical_not(missing_mask)))

    def test_normalized_features(self):
        self.gym_env.observation_space = BusConnectivityGraphObsSpace(self.env.observation_space, {
            "active_power_forecast": [-122.4, 116666],
            "active_power": [-122.4, 116666],
            "reactive_power_forecast": [-85.7, 107.228],
            "reactive_power": [-85.7, 107.228],
            "voltage": [-142.1, 142.1],
            "voltage_angle": [-14.212, 1.246],
            "current": [0, 1512],
            "rho": [0, 1]
        })
        obs, _ = self.gym_env.reset()
        self.assertTrue(np.all(obs[NODES] > -0.1))
        self.assertTrue(np.all(obs[NODES] < 1.2))
