import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm

from common import G2OpGymEnv, BusConnectivityGraphObsSpace, NODES, EDGE_INDEX
from nri import prior_from_env
from nri.agent.graphormer.GraphormerEncoder import GraphormerNRIEncoder
from torch.nn import functional as F


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

    def test_encoder_learns_prior(self):
        env = G2OpGymEnv(obs_space_creation=lambda e: BusConnectivityGraphObsSpace(e.observation_space))
        prior = prior_from_env(0.9, env)
        ds_size = 1
        node_features = []
        edge_indices = []
        outputs = []
        for _ in tqdm(range(ds_size), "Create Dataset"):
            node_features.append(torch.from_numpy(env.reset()[0][NODES]))
            edge_indices.append(torch.from_numpy(env.reset()[0][EDGE_INDEX]))
            outputs.append(prior)

        ds = TensorDataset(torch.stack(node_features, 0), torch.stack(edge_indices, 0), torch.stack(outputs, 0))
        loader = DataLoader(ds, batch_size=1, shuffle=True)
        encoder = GraphormerNRIEncoder(x_dim=env.observation_space.x_dim, hidden_dim=2, num_edge_types=2, max_degree=8, max_path_distance=9)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.1)

        for epoch in range(500):
            total_loss = 0.0
            for x, edge_index, label in loader:
                optimizer.zero_grad()
                B, N, X_dim = x.shape
                E = N * (N - 1)
                x = x.reshape(B * N, -1)
                batch = torch.arange(B).repeat_interleave(N)
                edge_logits = encoder.forward(x, batch=batch, powerline_edge_index=edge_index[0])
                edge_probs = F.softmax(edge_logits, dim=-1).reshape(B, E, 2)
                loss = (edge_probs * (torch.log(edge_probs + 0.0000001) - torch.log(label + 0.0000001))).sum(dim=-1).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch}, Loss: {total_loss/len(ds)}")

        x = node_features[0]
        edge_logits = encoder.forward(x, powerline_edge_index=edge_indices[0])
        edge_probs = F.softmax(edge_logits, dim=-1).detach().cpu().numpy()
        self.assertLessEqual(np.abs(edge_probs - prior.numpy()).max(), 0.01) # all probs within 10% of prior
        from visualization import visualize_graph, PlottingArgs, get_node_styles, visualize_posterior
        visualize_graph(PlottingArgs(
            env.observation_space.num_nodes,
            get_node_styles(env._g2op_env, env.observation_space.__class__),
            env.reset()[0][EDGE_INDEX],
            latent_edge_probs=edge_probs,
        )).show()

        visualize_posterior(edge_probs, prior).show()
