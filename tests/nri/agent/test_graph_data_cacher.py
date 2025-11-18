import unittest

import torch
from torch import Tensor
from torch_geometric.data import Data

from nri.agent.graphormer.graph_data_cacher import GraphDataRetriever


class TestGraphDataCacher(unittest.TestCase):
    def setUp(self):
        self.data_retriever = GraphDataRetriever()
        self.edge_index = torch.Tensor([[0, 1], [1, 2], [2, 3], [1, 3]]).T.to(torch.long)
        self.graph_data = Data(num_nodes=5, edge_index=self.edge_index)

    def test_retrieve_graph_data(self):
        graph_data = self.data_retriever.get(self.graph_data)
        self.assertEqual(graph_data.num_nodes, 5)
        self.assertEqual(graph_data.edge_index.shape, (2, 4))
        self.assertEqual(graph_data.in_degree.shape, (5, ))
        self.assertEqual(graph_data.out_degree.shape, (5, ))
        self.assertEqual(graph_data.node_paths_length.shape, (5, 5))
        torch.testing.assert_close(graph_data.in_degree, Tensor([0, 1, 1, 2, 0]).to(dtype=torch.long))
        torch.testing.assert_close(graph_data.out_degree, Tensor([1, 2, 1, 0, 0]).to(dtype=torch.long))
        torch.testing.assert_close(graph_data.node_paths_length, Tensor([
            [1, 2, 3, 3, 0],
            [0, 1, 2, 2, 0],
            [0, 0, 1, 2, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ]).to(dtype=torch.long))
