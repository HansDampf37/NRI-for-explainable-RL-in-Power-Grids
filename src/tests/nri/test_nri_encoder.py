import unittest

import torch

from src.nri.Encoder import Encoder


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
            num_edge_types=self.e_out_dim,
            dropout_prob=0.1
        )

    def test_forward(self):
        with torch.no_grad():
            output = self.encoder.forward(self.x)
        num_fully_connected_edges = self.num_nodes ** 2 - self.num_nodes
        self.assertEqual(output.shape, (self.number_traj, num_fully_connected_edges, self.e_out_dim))

    def test_forward_with_edge_index(self):
        with torch.no_grad():
            output = self.encoder.forward(self.x, self.edge_index)
        self.assertEqual(output.shape, (self.number_traj, self.num_edges, self.e_out_dim))

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        x = self.x.clone().requires_grad_(True)
        out = self.encoder.forward(x, self.edge_index)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_forward_shape_unbatched(self):
        num_fully_connected_edges = self.num_nodes ** 2 - self.num_nodes
        out_target_shape = (num_fully_connected_edges, self.e_out_dim)
        with torch.no_grad():
            out = self.encoder.forward(self.x[0])
        self.assertEqual(out.shape, out_target_shape)

