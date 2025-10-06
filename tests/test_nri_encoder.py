import unittest

import torch

from nri.Encoder import Encoder


class TestNRIEncoder(unittest.TestCase):
    def setUp(self):
        self.number_traj = 100
        self.trajectory_length = 10

        self.num_nodes = 60
        self.num_edges = 70

        self.node_dim = 20
        self.hidden_dim = 10
        self.e_out_dim = 2 # 2 class discrete distribution per edge (present vs not present)

        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.x = torch.randn(self.number_traj, self.trajectory_length, self.num_nodes, self.node_dim)

        self.encoder = Encoder(
            x_dim=self.node_dim,
            trajectory_length=self.trajectory_length,
            hidden_dim=self.hidden_dim,
            e_out_dim=self.e_out_dim,
            dropout_prob=0.1
        )

    def test_forward(self):
        with torch.no_grad():
            output = self.encoder.forward(self.x)
        num_fully_connected_edges = self.num_nodes ** 2 - self.num_nodes
        self.assertEqual(output.shape, (self.number_traj, num_fully_connected_edges, self.e_out_dim))
