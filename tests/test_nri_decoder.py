import unittest
import torch
from nri.Decoder import Decoder


class TestDecoder(unittest.TestCase):
    def setUp(self):
        # general setup
        self.batch_size = 8
        self.num_nodes = 12
        self.num_edges = 20
        self.node_dim = 16
        self.hidden_dim = 32
        self.msg_hid = 24
        self.msg_out = 12
        self.num_edge_types = 2

        # random edge index [2, E]
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))

        # node features: [B, N, F]
        self.x = torch.randn(self.batch_size, self.num_nodes, self.node_dim)

        # edge type probabilities: [B, E, edge_types]
        self.edge_types = torch.rand(self.batch_size, self.edge_index.size(1), self.num_edge_types)
        self.edge_types /= self.edge_types.sum(dim=-1, keepdim=True)

        # instantiate decoder
        self.decoder = Decoder(
            x_dim=self.node_dim,
            num_edge_types=self.num_edge_types,
            hidden_dim=self.hidden_dim,
            dropout_prob=0.3,
            skip_first=False,
        )

    def test_forward_shape(self):
        """Check that single_step_forward returns output of same shape as input."""
        with torch.no_grad():
            out = self.decoder.single_step_forward(self.x, self.edge_types, self.edge_index)
        self.assertEqual(out.shape, self.x.shape)

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        x = self.x.clone().requires_grad_(True)
        out = self.decoder.single_step_forward(x, self.edge_types, self.edge_index)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_skip_first_edge_type(self):
        """Check that skip_first=True excludes first edge type."""
        dec_normal = self.decoder
        dec_skip = Decoder(
            x_dim=self.node_dim,
            num_edge_types=self.num_edge_types,
            hidden_dim=self.hidden_dim,
            dropout_prob=0.0,
            skip_first=True,
        )

        with torch.no_grad():
            out_normal = dec_normal.single_step_forward(self.x, self.edge_types, self.edge_index)
            out_skip = dec_skip.single_step_forward(self.x, self.edge_types, self.edge_index)

        # Should differ since one ignores edge type 0
        self.assertFalse(torch.allclose(out_normal, out_skip))
