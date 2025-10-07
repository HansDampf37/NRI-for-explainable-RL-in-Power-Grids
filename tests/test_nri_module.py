import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from nri.ElboObjective import ElboLoss
from nri.NRI import NRIModule


class TestNRIModule(unittest.TestCase):
    def setUp(self):
        # general setup
        self.batch_size = 8
        self.num_nodes = 12
        self.num_edges = 20
        self.x_dim = 4
        self.hidden_dim = 32
        self.num_edge_types = 2
        self.trajectory_length = 100

        # random edge index [2, E]
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))

        # node features: [B, T, N, X_dim]
        self.x = torch.randn(self.batch_size, self.trajectory_length, self.num_nodes, self.x_dim)

        # instantiate module
        self.module = NRIModule(
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            trajectory_length=self.trajectory_length,
            num_edge_types=self.num_edge_types,
            pred_steps=4,
            dropout_prob=0.1,
        )

    def test_differentiable(self):
        """Check that gradients can flow through the output."""
        x = self.x.clone().requires_grad_(True)
        predictions, latent_edges = self.module(x)
        loss = predictions.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_forward_shape(self):
        with torch.no_grad():
            predictions, latent_edges = self.module.forward(self.x)
        self.assertEqual(predictions.shape, self.x.shape)

    def test_training_converges(self):
        adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_edges):
            src, target = self.edge_index[:, i]
            adj_matrix[src, target] = 1.0
        normalizer = adj_matrix.sum(dim=1, keepdim=True)
        normalizer[normalizer == 0] = 1
        adj_matrix /= normalizer

        def create_dataset():
            trajectories = []
            for _ in range(1000):
                x_t = torch.randn(self.num_nodes, self.x_dim)
                trajectory = [x_t]
                for _ in range(1, self.trajectory_length):
                    x_t = adj_matrix @ x_t
                    trajectory.append(x_t)
                trajectories.append(torch.stack(trajectory, dim=0))
            return TensorDataset(torch.stack(trajectories, dim=0))

        ds = create_dataset()
        dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        prior = np.array([0, 1])
        criterion = ElboLoss(prior)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=0.01)
        self.module.train()
        num_epochs = 0 # TODO increase again
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in dataloader:
                batch = batch[0]
                optimizer.zero_grad()
                predictions, latent_edges = self.module.forward(batch, self.edge_index)
                target = batch[:, 1:, :, :]
                loss = criterion(predictions, target, latent_edges)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch}, loss {running_loss}")

        # see latent edges

        batch = next(iter(dataloader))[0]
        sampled_edges = self.module.gumbel_softmax(self.module.encoder(batch, self.edge_index), hard=True)
        print(sampled_edges)
        latent_edges = self.edge_index[:, sampled_edges[:, :, 1] == 1]
        print(latent_edges)
