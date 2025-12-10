import unittest

import grid2op
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
from src.visualization.utils import get_node_styles, visualize_graph, latent_edge_hist, PlottingArgs, visualize_posterior


class TestVisualization(unittest.TestCase):
    def test_get_node_positions(self):
        env = grid2op.make("l2rpn_case14_sandbox")
        obs_space = BusConnectivityGraphObsSpace
        node_styles = get_node_styles(env, obs_space)
        self.assertEqual(node_styles[0].position.shape, (2, ))
        x = [node_style.position[0] for node_style in node_styles]
        y = [node_style.position[1] for node_style in node_styles]
        c = [node_style.color for node_style in node_styles]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=c, s=20)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Node Positions")
        plt.axis('equal')  # keeps aspect ratio square
        plt.grid(True)
        # plt.show()
        plt.close()

    def test_visualize_latent_graph(self):
        env = grid2op.make("l2rpn_case14_sandbox")
        obs_space = BusConnectivityGraphObsSpace(env.observation_space)
        obs = env.reset()

        num_edge_types = 3
        num_edges_fully_connected = obs_space.num_nodes * (obs_space.num_nodes - 1)

        edge_type_probs = torch.randn((num_edges_fully_connected, num_edge_types))
        edge_type_probs[edge_type_probs < 2.5] = 0.0
        edge_type_probs /= edge_type_probs.sum(dim=1, keepdim=True)

        plotting_args = PlottingArgs(
            num_nodes=obs_space.num_nodes,
            node_styles=get_node_styles(env, obs_space.__class__),
            powerline_edge_index=obs_space.to_gym(obs)[EDGE_INDEX],
            latent_edge_probs=edge_type_probs.numpy(),
            latent_edge_weight=2,
            skip_last_edge_type=True
        )
        fig = visualize_graph(plotting_args)
        # plt.show()
        plt.close(fig)

    def test_visualize_posterior(self):
        num_edges = 1000

        prior = 0.1 * np.random.randn(num_edges, ) + 0.8
        posterior = 0.1 * np.random.randn(num_edges, ) + 0.6

        prior = np.stack([prior, 1 - prior], axis=-1)
        posterior = np.stack([posterior, 1 - posterior], axis=-1)

        fig = visualize_posterior(posterior, prior)
        # plt.show()
        plt.close(fig)


    def test_visualize_edge_hist(self):
        num_nodes = 57
        num_edge_types = 3
        num_edges_fully_connected = num_nodes * (num_nodes - 1)

        edge_type_probs = torch.rand((num_edges_fully_connected, num_edge_types)) + 0.0001
        edge_type_probs /= edge_type_probs.sum(dim=-1, keepdim=True)

        fig = latent_edge_hist(edge_type_probs)
        # plt.show()
        plt.close(fig)


