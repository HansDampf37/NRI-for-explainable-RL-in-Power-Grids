import unittest

import torch

from common.MLP import MLP


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.output_dim = 30
        self.hidden_dim = 20
        self.mlp = MLP(self.input_dim, self.hidden_dim, self.output_dim)
        self.mlp.eval()

    def test_forward(self):
        num_trajectories = 100
        num_nodes = 50
        input_data = torch.randn(num_trajectories, num_nodes, self.input_dim)
        output_data = self.mlp.forward(input_data)
        self.assertEqual(output_data.size(), torch.Size([num_trajectories, num_nodes, self.output_dim]))


