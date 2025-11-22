import unittest

import numpy as np
import torch

from common.env import G2OpGymEnv
from common.graph_structured_observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
from nri.utils import prior_from_env
from ra_agents.Encoder import Encoder
from torch.nn import functional as F
from ra_agents.pretrain_encoder import GraphDataset, create_dataset, train


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

        ds: GraphDataset = create_dataset(env, prior, 1)
        ds_test: GraphDataset = create_dataset(env, prior, 1)
        encoder = Encoder(x_dim=env.observation_space.x_dim, hidden_dim=2, num_edge_types=2, dropout_prob=0.0)
        train(encoder=encoder, ds=ds, num_epochs=2000)

        x = ds_test.get(0).x
        edge_index = ds_test.get(0).edge_index
        edge_logits = encoder.forward(x, powerline_edge_index=edge_index)
        edge_probs = F.softmax(edge_logits, dim=-1).detach().cpu().numpy()
        self.assertLessEqual(np.abs(edge_probs - prior.numpy()).max(), 0.05)  # all probs within 5% of prior

        visualize = False
        if visualize:
            from visualization import visualize_graph, PlottingArgs, get_node_styles, visualize_posterior
            visualize_graph(PlottingArgs(
                env.observation_space.num_nodes,
                get_node_styles(env._g2op_env, env.observation_space.__class__),
                env.reset()[0][EDGE_INDEX],
                latent_edge_probs=edge_probs,
            )).show()
            visualize_posterior(edge_probs, prior).show()
