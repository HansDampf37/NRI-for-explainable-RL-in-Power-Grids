import unittest

import numpy as np
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader

from common.GNN import NodeEdgeMP, GNNFeatureExtractor


class TestMessagePassing(unittest.TestCase):
    def setUp(self):
        self.num_node_features = 11
        self.num_edge_features = 12
        self.num_nodes = 10
        self.num_edges = 20
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))

        self.node_features = torch.randn(self.num_nodes, self.num_node_features)
        self.edge_features = torch.randn(self.num_edges, self.num_edge_features)

    def test_message_passing(self):
        num_node_features_out = 1
        num_edge_features_out = 2
        mp = NodeEdgeMP(
            x_dim=self.num_node_features,
            e_dim=self.num_edge_features,
            x_out_dim=num_node_features_out,
            e_out_dim=num_edge_features_out,
        )
        node_features, edge_features = mp(self.node_features, self.edge_features, self.edge_index)
        self.assertEqual(node_features.shape, torch.Size([self.num_nodes, num_node_features_out]))
        self.assertEqual(edge_features.shape, torch.Size([self.num_edges, num_edge_features_out]))

    def test_no_residual(self):
        kwargs = dict(
            x_dim=self.num_node_features,
            e_dim=self.num_edge_features,
            x_out_dim=self.num_node_features + 1,
            e_out_dim=self.num_edge_features + 1,
            residual=True,
        )
        self.assertRaises(ValueError, NodeEdgeMP, **kwargs)

    def test_residual(self):
        NodeEdgeMP(
            x_dim=self.num_node_features,
            e_dim=self.num_edge_features,
            x_out_dim=self.num_node_features,
            e_out_dim=self.num_edge_features,
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

        dataset = KarateClub()  # single graph, 34 nodes, labels per node
        data = dataset[0]
        data.e = torch.ones(data.edge_index.size(1), 4)  # dummy edge features, dim=4
        data.y = Tensor(np.array(2.0))
        dataloader = DataLoader([data])

        model = GNNFeatureExtractor(
            x_dim=data.x.size(-1),
            e_dim=data.e.size(-1),
            x_hidden_dim=32,
            e_hidden_dim=32,
            x_out_dim=1,
            e_out_dim=1,
            n_layers=3,
        )

        opt = optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        first_loss = None
        last_loss = None
        for epoch in range(200):
            running_loss = 0.0
            for batch in dataloader:
                x, edge_index, y, _, e, batch_index, _ = batch
                logits = model(x[1], e[1], edge_index[1], batch_index[1])[:, :dataset.num_classes]  # [1, num_classes]
                loss = loss_fn(logits.sum(), y[1])
                running_loss += loss.item()

                opt.zero_grad()
                loss.backward()
                opt.step()

                first_loss = first_loss or loss.item()
                last_loss = loss.item()

            print(f"Epoch {epoch:02d} | Loss {running_loss:.4f}")
        self.assertTrue(last_loss < first_loss * 0.2)
