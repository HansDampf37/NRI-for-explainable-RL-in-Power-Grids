import unittest
import torch
from nri.nri_rl import NRI_GNN

class TestNRIFeatureExtractor(unittest.TestCase):
    def setUp(self):
        # Initialize parameters for NRIFeatureExtractor
        self.x_dim = 5
        self.hidden_dim = 10
        self.x_out_dim = 3
        self.num_edge_types = 4
        self.dropout_prob = 0.2

        self.model = NRI_GNN(
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            x_out_dim=self.x_out_dim,
            num_edge_types=self.num_edge_types,
            dropout_prob=self.dropout_prob
        )

        # Dummy data for testing
        self.batch_size = 2
        self.num_nodes = 6
        self.edge_index = torch.randint(0, self.num_nodes, (2, 7))  # Random edge indices
        self.node_features = torch.rand(self.num_nodes, self.x_dim)  # Random node feature tensor
        self.batch = torch.tensor([0, 0, 1, 1, 0, 1])  # Batch tensor

    def test_forward_shape(self):
        predictions, p_z_given_x = self.model(self.node_features, self.edge_index, self.batch)
        # Check output shapes
        self.assertEqual(predictions.shape, (self.batch_size, self.x_out_dim), "Output shape mismatch")
        self.assertEqual(p_z_given_x.shape, (self.edge_index.size(1), self.num_edge_types), "Softmax probabilities shape mismatch")

    def test_forward_values(self):
        predictions, p_z_given_x = self.model(x=self.node_features, batch=self.batch)
        # Check that the softmax outputs are valid probabilities
        self.assertTrue(torch.all(p_z_given_x >= 0) and torch.all(p_z_given_x <= 1), "Probabilities are out of bounds")
        self.assertTrue(torch.allclose(p_z_given_x.sum(dim=-1), torch.ones(p_z_given_x.size(0))), "Softmax probabilities do not sum to 1")

    def test_inference(self):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions, p_z_given_x = self.model(self.node_features, self.edge_index, self.batch)
            # Check that predictions are valid after inference
            self.assertIsNotNone(predictions, "Predictions should not be None")
            self.assertIsNotNone(p_z_given_x, "Posterior probabilities should not be None")

if __name__ == '__main__':
    unittest.main()
