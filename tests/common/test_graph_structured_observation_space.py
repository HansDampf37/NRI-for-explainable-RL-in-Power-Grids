import unittest

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
