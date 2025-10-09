import unittest
import torch
from nri.Decoder import Decoder
from nri.utils import fully_connected_edge_index


class TestDecoder(unittest.TestCase):
    def setUp(self):
        # general setup
        self.batch_size = 8
        self.num_nodes = 12
        self.edge_index = fully_connected_edge_index(self.num_nodes, "cpu", False)
        self.num_edges = len(self.edge_index[0])
        self.x_dim = 16
        self.hidden_dim = 32
        self.num_edge_types = 2
        self.trajectory_length = 100
        self.prediction_length = self.trajectory_length - 1

        # node features: [B, T, N, F]
        self.x = torch.randn(self.batch_size, self.trajectory_length, self.num_nodes, self.x_dim)

        # edge type probabilities: [B, E, edge_types]
        self.edge_types = torch.rand(self.batch_size, self.num_edges, self.num_edge_types)
        self.edge_types /= self.edge_types.sum(dim=-1, keepdim=True)

        # instantiate decoder
        self.decoder = Decoder(
            x_dim=self.x_dim,
            num_edge_types=self.num_edge_types,
            hidden_dim=self.hidden_dim,
            dropout_prob=0.3,
            skip_first=False,
        )

    def test_single_step_forward_shape(self):
        """Check that single_step_forward returns output of same shape as input."""
        with torch.no_grad():
            x = self.x[:, [0], :, :]
            out = self.decoder.single_step_forward(x, self.edge_types, self.edge_index)
        self.assertEqual(out.shape, x.shape)

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        x = self.x[:,[0],:,:].clone().requires_grad_(True)
        out = self.decoder.single_step_forward(x, self.edge_types, self.edge_index)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_forward_shape(self):
        out_target_shape = (self.batch_size, self.prediction_length, self.num_nodes, self.x_dim)
        with torch.no_grad():
            out = self.decoder.forward(self.x, self.edge_types, self.edge_index, pred_steps=4)
        self.assertEqual(out.shape, out_target_shape)

    def test_forward_with_pred_steps_modulo_1(self):
        out_target_shape = (self.batch_size, self.prediction_length, self.num_nodes, self.x_dim)
        with torch.no_grad():
            # 100 timesteps cannot be split into chunks of size 7
            out = self.decoder.forward(self.x, self.edge_types, self.edge_index, pred_steps=7)
        self.assertEqual(out.shape, out_target_shape)

    def test_forward_shape_unbatched(self):
        out_target_shape = (self.prediction_length, self.num_nodes, self.x_dim)
        with torch.no_grad():
            out = self.decoder.forward(self.x[0], self.edge_types[0], self.edge_index, pred_steps=4)
        self.assertEqual(out.shape, out_target_shape)

