import unittest

import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from common import G2OpGymEnv, BusConnectivityGraphObsSpace, NODES, EDGE_INDEX
from nri.agent.Encoder import Encoder
from nri.utils import prior_from_env


class TestEncoder(unittest.TestCase):
    def test_forward_with_provided_edge_index(self):
        x_dim = 4
        hidden_dim = 6
        K = 3
        enc = Encoder(x_dim=x_dim, hidden_dim=hidden_dim, num_edge_types=K)
        N = 3
        x = torch.randn(N, x_dim)
        # construct fully connected directed edge_index without self loops
        senders = []
        receivers = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    senders.append(i)
                    receivers.append(j)
        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        out = enc.forward(x, edge_set=edge_index)
        # output should have shape [E, K]
        self.assertEqual(out.shape, (len(senders), K))
        self.assertFalse(torch.isnan(out).any())

    def test_forward_with_batch_default_edges(self):
        x_dim = 2
        num_edge_types = 3
        enc = Encoder(x_dim=x_dim, hidden_dim=4, num_edge_types=num_edge_types)
        # 3 nodes split in two graphs: graph 0 has 2 nodes, graph 1 has 1 node
        num_nodes = 3
        x = torch.randn(num_nodes, x_dim)
        batch = torch.tensor([0, 0, 1], dtype=torch.long)
        out = enc.forward(x, edge_set=None, batch=batch)
        # graph0 -> 2 nodes => 2 directed edges (without self loops), graph1 -> 1 node => 0 edges
        num_edges = 2
        self.assertEqual(out.size(0), num_edges)
        self.assertEqual(out.size(1), num_edge_types)
        self.assertFalse(torch.isnan(out).any())


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
        encoder = Encoder(x_dim=env.observation_space.x_dim, hidden_dim=2, num_edge_types=2, dropout_prob=0.0)
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
                edge_probs = f.softmax(edge_logits, dim=-1).reshape(B, E, 2)
                loss = (edge_probs * (torch.log(edge_probs + 0.0000001) - torch.log(label + 0.0000001))).sum(dim=-1).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch}, Loss: {total_loss/len(ds)}")

        x = node_features[0]
        edge_logits = encoder.forward(x, powerline_edge_index=edge_indices[0])
        edge_probs = f.softmax(edge_logits, dim=-1).detach().cpu().numpy()
        self.assertLessEqual(np.abs(edge_probs - prior.numpy()).max(), 0.1) # all probs within 10% of prior
        # from visualization.utils import visualize_graph, PlottingArgs, get_node_styles, visualize_posterior
        # visualize_graph(PlottingArgs(
        #     env.observation_space.num_nodes,
        #     get_node_styles(env._g2op_env, env.observation_space.__class__),
        #     env.reset()[0][EDGE_INDEX],
        #     latent_edge_probs=edge_probs,
        # )).show()

        # visualize_posterior(edge_probs, prior).show()
