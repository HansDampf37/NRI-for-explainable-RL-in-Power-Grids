import unittest

import numpy as np
import torch

from ra_agents.RAFeatureExtractor import RAFeatureExtractor


class TestRAFeatureExtractor(unittest.TestCase):
    def setUp(self):
        # Initialize parameters for NRIFeatureExtractor
        self.x_dim = 5
        self.hidden_dim = 10
        self.x_out_dim = 3
        self.num_edge_types = 4
        self.dropout_prob = 0.2

        self.model = RAFeatureExtractor(
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            x_out_dim=self.x_out_dim,
            num_edge_types=self.num_edge_types,
            num_layers=3,
            dropout_prob=self.dropout_prob
        )

        # Dummy data for testing
        self.batch_size = 2
        self.num_nodes_per_batch = 3
        self.node_features = torch.rand(self.num_nodes_per_batch * self.batch_size, self.x_dim)  # Random node feature tensor
        self.batch = torch.tensor(np.array([[i] * self.num_nodes_per_batch for i in range(self.batch_size)]).flatten())  # Batch tensor
        self.num_fc_edges_per_batch = self.num_nodes_per_batch * (self.num_nodes_per_batch - 1)

    def test_forward_shape(self):
        predictions, p_z_given_x = self.model(self.node_features, batch=self.batch)
        # Check output shapes
        self.assertEqual(predictions.shape, (self.batch_size, self.x_out_dim))
        self.assertEqual(p_z_given_x.shape, (self.batch_size, self.num_fc_edges_per_batch, self.num_edge_types))

    def test_forward_values(self):
        predictions, p_z_given_x = self.model(x=self.node_features, batch=self.batch)
        # Check that the softmax outputs are valid probabilities
        self.assertTrue(torch.all(p_z_given_x >= 0) and torch.all(p_z_given_x <= 1), "Probabilities are out of bounds")
        self.assertTrue(torch.allclose(p_z_given_x.sum(dim=-1), torch.ones(self.batch_size, p_z_given_x.size(1))), "Softmax probabilities do not sum to 1")

    def test_inference(self):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions, p_z_given_x = self.model(self.node_features, batch=self.batch)
            # Check that predictions are valid after inference
            self.assertIsNotNone(predictions, "Predictions should not be None")
            self.assertIsNotNone(p_z_given_x, "Posterior probabilities should not be None")
