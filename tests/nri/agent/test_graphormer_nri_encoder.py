import unittest

import torch

from nri.agent.graphormer.GraphormerEncoder import GraphormerNRIEncoder


class TestGraphormerNRIEncoder(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.E = 100
        self.NUM_EDGE_TYPES = 2
        self.encoder = GraphormerNRIEncoder(8, 8, self.NUM_EDGE_TYPES, 8, 10)

    def test_forward(self):
        x = torch.randn((self.N, 8))
        edge_index = torch.randint(0, self.N, size=(2, self.E))
        output = self.encoder.forward(
            x=x, powerline_edge_index=edge_index
        )

        self.assertEqual(output.shape, (self.N * (self.N - 1), self.NUM_EDGE_TYPES))
