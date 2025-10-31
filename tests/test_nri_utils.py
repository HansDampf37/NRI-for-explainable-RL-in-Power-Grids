import logging
import unittest
from unittest.mock import create_autospec

import torch
from torch_geometric.utils import to_dense_adj

from nri.utils import fully_connected_edge_index, Edge2Node, Node2Edge, EdgeNode2Node, warn_large_loss, \
    fully_connected_edge_index_per_batch


class TestNRIUtils(unittest.TestCase):
    def test_create_edge_index(self):
        num_nodes = 10
        edge_index = fully_connected_edge_index(num_nodes, "cpu", False)
        self.assertEqual(edge_index.shape, (2, num_nodes ** 2 - num_nodes))
        edge_index = fully_connected_edge_index(num_nodes, "cpu", True)
        self.assertEqual(edge_index.shape, (2, num_nodes ** 2))

    def test_single_graph_no_self_loops(self):
        batch = torch.tensor([0, 0, 0])
        edge_index = fully_connected_edge_index_per_batch(batch, self_loops=False)
        adj = to_dense_adj(edge_index)[0]
        expected = torch.ones((3, 3), dtype=torch.bool)
        expected.fill_diagonal_(False)
        self.assertTrue(torch.equal(adj.bool(), expected))

    def test_single_graph_with_self_loops(self):
        batch = torch.tensor([0, 0, 0])
        edge_index = fully_connected_edge_index_per_batch(batch, self_loops=True)
        adj = to_dense_adj(edge_index)[0]
        expected = torch.ones((3, 3), dtype=torch.bool)
        self.assertTrue(torch.equal(adj.bool(), expected))

    def test_multiple_graphs(self):
        batch = torch.tensor([0, 0, 1, 1, 1])
        edge_index = fully_connected_edge_index_per_batch(batch, self_loops=False)
        adj = to_dense_adj(edge_index, batch=batch)
        expected_0 = torch.ones((2, 2), dtype=torch.bool)
        expected_0.fill_diagonal_(False)
        expected_1 = torch.ones((3, 3), dtype=torch.bool)
        expected_1.fill_diagonal_(False)
        self.assertTrue(torch.equal(adj[0].bool()[:2, :2], expected_0))
        self.assertTrue(torch.equal(adj[1].bool(), expected_1))

    def test_empty_graph(self):
        batch = torch.tensor([0, 1, 1])
        edge_index = fully_connected_edge_index_per_batch(batch, self_loops=False)
        adj = to_dense_adj(edge_index, batch=batch)
        expected_0 = torch.zeros((2, 2), dtype=torch.bool)
        expected_1 = torch.ones((2, 2), dtype=torch.bool)
        expected_1.fill_diagonal_(False)
        self.assertTrue(torch.equal(adj[0].bool(), expected_0))
        self.assertTrue(torch.equal(adj[1].bool(), expected_1))

    def test_device_cpu(self):
        batch = torch.tensor([0, 0])
        edge_index = fully_connected_edge_index_per_batch(batch, device="cpu")
        self.assertEqual(edge_index.device.type, "cpu")

    def test_warn_large_loss(self):
        num_features = 8
        nri_prediction = torch.rand(10, 10, 10, num_features)
        nri_target = torch.rand(10, 10, 10, num_features)
        logger = create_autospec(logging.Logger, instance=True)
        per_feature_mse = warn_large_loss(logger, nri_prediction, nri_target)
        self.assertEqual(per_feature_mse.shape, (num_features,))

class TestEdge2Node(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.edge_index = fully_connected_edge_index(self.num_nodes, "cpu", False)
        self.num_edges = len(self.edge_index[0])
        self.e_dim = 4
        self.hidden_dim = 8
        self.x_dim = 6
        self.edge2node = Edge2Node(self.e_dim, self.hidden_dim, self.x_dim, dropout_prob=0.1)
        self.e = torch.randn(self.num_edges, self.e_dim)

    def test_forward(self):
        out_x = self.edge2node(self.e, self.edge_index)
        self.assertEqual(out_x.shape, (self.num_nodes, self.x_dim))

    def test_batched_forward(self):
        batch_size = 8
        batched_e = torch.stack([self.e] * batch_size, dim=0)
        out_x = self.edge2node(batched_e, self.edge_index)
        self.assertEqual(out_x.shape, (batch_size, self.num_nodes, self.x_dim))

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        e = self.e.clone().requires_grad_(True)
        out = self.edge2node.forward(e, self.edge_index)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(e.grad)
        self.assertEqual(e.grad.shape, e.shape)

class TestNode2Edge(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.edge_index = fully_connected_edge_index(self.num_nodes, "cpu", False)
        self.num_edges = len(self.edge_index[0])
        self.e_dim = 4
        self.hidden_dim = 8
        self.x_dim = 6
        self.node2edge = Node2Edge(self.x_dim, self.hidden_dim, self.e_dim, dropout_prob=0.1)
        self.x = torch.randn(self.num_nodes, self.x_dim)

    def test_forward(self):
        out_e = self.node2edge(self.x, self.edge_index)
        self.assertEqual(out_e.shape, (self.num_edges, self.e_dim))

    def test_batched_forward(self):
        batch_size = 8
        batched_x = torch.stack([self.x] * batch_size, dim=0)
        out_e = self.node2edge(batched_x, self.edge_index)
        self.assertEqual(out_e.shape, (batch_size, self.num_edges, self.e_dim))

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        x = self.x.clone().requires_grad_(True)
        out = self.node2edge.forward(x, self.edge_index)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

class TestEdgeNode2Node(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.edge_index = fully_connected_edge_index(self.num_nodes, "cpu", False)
        self.num_edges = len(self.edge_index[0])
        self.e_dim = 4
        self.hidden_dim = 8
        self.x_dim = 6
        self.x_out_dim = 8
        self.edge2node = EdgeNode2Node(self.x_dim, self.e_dim, self.hidden_dim, self.x_out_dim, dropout_prob=0.1)
        self.x = torch.randn(self.num_nodes, self.x_dim)
        self.e = torch.randn(self.num_edges, self.e_dim)

    def test_forward(self):
        out_x = self.edge2node(self.x, self.e, self.edge_index)
        self.assertEqual(out_x.shape, (self.num_nodes, self.x_out_dim))

    def test_batched_forward(self):
        batch_size = 8
        batched_e = torch.stack([self.e] * batch_size, dim=0)
        batched_x = torch.stack([self.x] * batch_size, dim=0)
        out_x = self.edge2node(batched_x, batched_e, self.edge_index)
        self.assertEqual(out_x.shape, (batch_size, self.num_nodes, self.x_out_dim))

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        x = self.x.clone().requires_grad_(True)
        e = self.e.clone().requires_grad_(True)
        out = self.edge2node.forward(x, e, self.edge_index)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(e.grad)
        self.assertIsNotNone(x.grad)
        self.assertEqual(e.grad.shape, e.shape)
        self.assertEqual(x.grad.shape, x.shape)

