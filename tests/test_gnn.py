import unittest

import torch

from common.GNN import MessagePassing, GNNFeatureExtractor


class TestMessagePassing(unittest.TestCase):
    def setUp(self):
        self.num_node_features = 11
        self.num_edge_features = 12
        self.num_nodes = 10
        self.num_edges = 20
        self.batch_size = 4
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.node_batch = torch.randint(0, self.batch_size, (self.num_nodes,))

        self.node_features = torch.randn(self.num_nodes, self.num_node_features)
        self.edge_features = torch.randn(self.num_edges, self.num_edge_features)

    def test_message_passing(self):
        num_node_features_out = 1
        num_edge_features_out = 2
        mp = MessagePassing(
            x_dim=self.num_node_features,
            e_dim=self.num_edge_features,
            output_x_dim=num_node_features_out,
            output_e_dim=num_edge_features_out,
        )
        node_features, edge_features = mp(self.node_features, self.edge_features, self.edge_index)
        self.assertEqual(node_features.shape, torch.Size([self.num_nodes, num_node_features_out]))
        self.assertEqual(edge_features.shape, torch.Size([self.num_edges, num_edge_features_out]))

    def test_no_residual(self):
        kwargs = dict(
            x_dim=self.num_node_features,
            e_dim=self.num_edge_features,
            output_x_dim=self.num_node_features + 1,
            output_e_dim=self.num_edge_features + 1,
            residual=True,
        )
        self.assertRaises(ValueError, MessagePassing, **kwargs)

    def test_residual(self):
        MessagePassing(
            x_dim=self.num_node_features,
            e_dim=self.num_edge_features,
            output_x_dim=self.num_node_features,
            output_e_dim=self.num_edge_features,
            residual=True,
        )

class TestGnnFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.num_node_features = 11
        self.num_edge_features = 12
        self.num_nodes = 10
        self.num_edges = 20
        self.batch_size = 4
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.node_batch = torch.randint(0, self.batch_size, (self.num_nodes,))
        self.node_features = torch.randn(self.num_nodes, self.num_node_features)
        self.edge_features = torch.randn(self.num_edges, self.num_edge_features)

    def test_gnn(self):
        output_size = 2
        fe = GNNFeatureExtractor(
            self.num_node_features,
            self.num_edge_features,
            128,
            128,
            output_size,
            output_size,
            3,
            0.1,
            True
        )
        output = fe(self.node_features, self.edge_features, self.edge_index, self.node_batch)
        self.assertEqual(output.shape, torch.Size([self.batch_size, output_size + output_size]))

    def test_parameter_count(self):
        output_size = 32
        hidden_size = 32
        fe = GNNFeatureExtractor(
            self.num_node_features,
            self.num_edge_features,
            hidden_size,
            hidden_size,
            output_size,
            output_size,
            3,
            0.1,
            True
        )
        print("Amount of trainable parameters:", sum(p.numel() for p in fe.parameters() if p.requires_grad))

    def test_convergence(self):
        """
        train on easy task and see that loss decreases
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch_geometric.datasets import KarateClub
        from torch_geometric.utils import to_undirected

        dataset = KarateClub()  # single graph, 34 nodes, labels per node
        data = dataset[0]

        # Prepare inputs
        x = data.x.float()  # [N, node_features]
        edge_index = to_undirected(data.edge_index)  # [2, E]
        num_nodes = x.size(0)
        e = torch.ones(edge_index.size(1), 4)  # dummy edge features, dim=4
        node_batch_indices = torch.zeros(num_nodes, dtype=torch.long)

        # Target = graph label (for test we take the majority of node labels)
        y = data.y.mode()[0].unsqueeze(0)  # just to have a graph-level label

        model = GNNFeatureExtractor(
            x_dim=x.size(-1),
            e_dim=e.size(-1),
            hidden_x_dim=32,
            hidden_edge=32,
            node_out_dim=4,
            edge_out_dim=0,
            n_layers=3,
        )

        opt = optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        first_loss = None
        last_loss = None
        for epoch in range(200):
            opt.zero_grad()
            logits = model(x, e, edge_index, node_batch_indices)[:, :dataset.num_classes]  # [1, num_classes]
            loss = loss_fn(logits, y)
            first_loss = first_loss or loss.item()
            last_loss = loss.item()
            loss.backward()
            opt.step()
            print(f"Epoch {epoch:02d} | Loss {loss.item():.4f}")
        self.assertTrue(last_loss < first_loss * 0.2)
