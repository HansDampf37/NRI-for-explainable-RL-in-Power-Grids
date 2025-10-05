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
        self.edge_types = 3

        # random edge index (no self-loops)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))

        # node features: [B, N, F]
        self.x = torch.randn(self.batch_size, self.num_nodes, self.node_dim)

        # edge type probabilities: [B, E, edge_types]
        rel_type = torch.rand(self.batch_size, self.edge_index.size(1), self.edge_types)
        self.rel_type = rel_type / rel_type.sum(dim=-1, keepdim=True)  # normalize to probs

        # instantiate decoder
        self.decoder = Decoder(
            node_dim=self.node_dim,
            edge_types=self.edge_types,
            msg_hid=self.msg_hid,
            msg_out=self.msg_out,
            hidden_dim=self.hidden_dim,
            dropout_prob=0.3,
            skip_first=False,
        )

    def test_forward_shape(self):
        """Check that single_step_forward returns output of same shape as input."""
        with torch.no_grad():
            out = self.decoder.single_step_forward(self.x, self.edge_index, self.rel_type)
        self.assertEqual(out.shape, self.x.shape)

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        x = self.x.clone().requires_grad_(True)
        out = self.decoder.single_step_forward(x, self.edge_index, self.rel_type)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_skip_first_edge_type(self):
        """Check that skip_first=True excludes first edge type."""
        dec_normal = self.decoder
        dec_skip = Decoder(
            node_dim=self.node_dim,
            edge_types=self.edge_types,
            msg_hid=self.msg_hid,
            msg_out=self.msg_out,
            hidden_dim=self.hidden_dim,
            dropout_prob=0.0,
            skip_first=True,
        )

        with torch.no_grad():
            out_normal = dec_normal.single_step_forward(self.x, self.edge_index, self.rel_type)
            out_skip = dec_skip.single_step_forward(self.x, self.edge_index, self.rel_type)

        # Should differ since one ignores edge type 0
        self.assertFalse(torch.allclose(out_normal, out_skip))

    def test_device_movement(self):
        """Ensure model and tensors work on GPU if available."""
        if torch.cuda.is_available():
            decoder = self.decoder.to("cuda")
            x = self.x.to("cuda")
            rel_type = self.rel_type.to("cuda")
            edge_index = self.edge_index.to("cuda")

            with torch.no_grad():
                out = decoder.single_step_forward(x, edge_index, rel_type)
            self.assertEqual(out.device.type, "cuda")
            self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
