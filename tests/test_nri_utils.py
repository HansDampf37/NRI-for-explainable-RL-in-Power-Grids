import unittest

from nri.utils import fully_connected_edge_index


class TestNRIUtils(unittest.TestCase):
    def test_create_edge_index(self):
        num_nodes = 10
        edge_index = fully_connected_edge_index(num_nodes, False)
        self.assertEqual(edge_index.shape, (2, num_nodes ** 2 - num_nodes))
        edge_index = fully_connected_edge_index(num_nodes, True)
        self.assertEqual(edge_index.shape, (2, num_nodes ** 2))