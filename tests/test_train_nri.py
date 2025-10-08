import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from nri.NRI import NRIModule
from nri.train_nri import train


class TestTrainNRI(unittest.TestCase):
    def setUp(self):
        # general setup
        self.training_set_size = 100
        self.num_nodes = 10
        self.num_edges = 10
        self.x_dim = 1
        self.hidden_dim = 4
        self.num_edge_types = 2
        self.trajectory_length = 500

        # random edge index [2, E]
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))

        # node features: [B, T, N, X_dim]
        self.x = torch.randn(self.training_set_size, self.trajectory_length, self.num_nodes, self.x_dim)

        # instantiate module
        self.module = NRIModule(
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            trajectory_length=self.trajectory_length,
            num_edge_types=self.num_edge_types,
            pred_steps=4,
            dropout_prob=0.1,
        )

    def _create_dataset(self, size: int):
        adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_edges):
            src, target = self.edge_index[:, i]
            adj_matrix[src, target] = 1.0
        normalizer = adj_matrix.sum(dim=1, keepdim=True)
        normalizer[normalizer == 0] = 1
        adj_matrix /= normalizer

        trajectories = []
        for _ in range(size):
            x_t = torch.randn(self.num_nodes, self.x_dim)
            trajectory = [x_t]
            for _ in range(1, self.trajectory_length):
                x_t = adj_matrix @ x_t
                trajectory.append(x_t)
            trajectories.append(torch.stack(trajectory, dim=0))
        return TensorDataset(torch.stack(trajectories, dim=0))

    def test_training_converges(self):
        ds_train = self._create_dataset(self.training_set_size)
        ds_test = self._create_dataset(10)
        prior = np.array([0.01, 0.99]) # encourage to use every edge of the edge index since they are all used by the system
        train(self.module, ds_train, ds_test, prior, self.edge_index, None, 10) # doesn't fail
