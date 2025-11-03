import unittest

import torch

from nri.NRI import NRIModule


class TestNRIModule(unittest.TestCase):
    def setUp(self):
        # general setup
        self.batch_size = 8
        self.num_nodes = 12
        self.x_dim = 4
        self.hidden_dim = 32
        self.num_edge_types = 2
        self.trajectory_length = 100

        # node features: [B, T, N, X_dim]
        self.x = torch.randn(self.batch_size, self.trajectory_length, self.num_nodes, self.x_dim)

        # instantiate module
        self.module = NRIModule(
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            trajectory_length=self.trajectory_length,
            num_edge_types=self.num_edge_types,
            pred_steps=4,
            dropout_prob=0.1,
        )

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        x = self.x.clone().requires_grad_(True)
        predictions, latent_edges = self.module(x)
        loss = predictions.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_forward_shape(self):
        with torch.no_grad():
            predictions, latent_edges = self.module.forward(self.x)
        self.assertEqual(predictions.shape, (self.batch_size, self.trajectory_length - 1, self.num_nodes, self.x_dim))

    def test_forward_shape_unbatched(self):
        predictions_target_shape = (self.trajectory_length - 1, self.num_nodes, self.x_dim)
        latent_edges_target_shape = (self.num_nodes * (self.num_nodes - 1), self.num_edge_types)
        with torch.no_grad():
            predictions, latent_edges = self.module.forward(self.x[0])
        self.assertEqual(predictions.shape, predictions_target_shape)
        self.assertEqual(latent_edges.shape, latent_edges_target_shape)