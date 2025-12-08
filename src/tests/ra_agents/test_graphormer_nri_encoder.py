import unittest

import numpy as np
import torch
from torch.nn import functional as F

from src.common.env import G2OpGymEnv
from src.common.graph_structured_observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
from src.nri.utils import prior_from_env
from src.ra_agents.graphormer.GraphormerEncoder import GraphormerNRIEncoder
from src.ra_agents.pretrain_encoder import create_dataset, GraphDataset, train


class TestGraphormerNRIEncoder(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.E = 100
        self.x_dim=8
        self.NUM_EDGE_TYPES = 2

        self.encoder = GraphormerNRIEncoder(
            x_dim=self.x_dim,
            hidden_dim=8,
            num_layers=1,
            num_edge_types=self.NUM_EDGE_TYPES,
            max_degree=8,
            max_path_distance=9
        )

    def test_forward(self):
        x = torch.randn((self.N, self.x_dim))
        edge_index = torch.randint(0, self.N, size=(2, self.E))
        output = self.encoder.forward(
            x=x, powerline_edge_index=edge_index
        )

        self.assertEqual(output.shape, (self.N * (self.N - 1), self.NUM_EDGE_TYPES))

    def test_encoder_learns_prior(self):
        env = G2OpGymEnv(obs_space_creation=lambda e: BusConnectivityGraphObsSpace(e.observation_space))
        prior = prior_from_env(0.9, env)

        ds: GraphDataset = create_dataset(env, prior, 1)
        ds_test: GraphDataset = create_dataset(env, prior, 1)
        encoder = GraphormerNRIEncoder(
            x_dim=env.observation_space.x_dim,
            hidden_dim=2,
            num_layers=1,
            num_edge_types=2,
            max_degree=8,
            max_path_distance=9
        )
        train(encoder=encoder, ds=ds, num_epochs=2000)

        x = ds_test.get(0).x
        edge_index = ds_test.get(0).edge_index
        edge_logits = encoder.forward(x, powerline_edge_index=edge_index)
        edge_probs = F.softmax(edge_logits, dim=-1).detach().cpu().numpy()
        self.assertLessEqual(np.abs(edge_probs - prior.numpy()).max(), 0.05)  # all probs within 5% of prior

        visualize = False
        if visualize:
            from src.visualization import visualize_graph, PlottingArgs, get_node_styles, visualize_posterior
            visualize_graph(PlottingArgs(
                env.observation_space.num_nodes,
                get_node_styles(env._g2op_env, env.observation_space.__class__),
                env.reset()[0][EDGE_INDEX],
                latent_edge_probs=edge_probs,
            )).show()
            visualize_posterior(edge_probs, prior).show()
