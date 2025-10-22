import unittest

import grid2op
import torch
from matplotlib import pyplot as plt

from common.graph_structured_observation_space import BusConnectionsGraphObsSpace, EDGE_INDEX
from visualization.utils import get_node_positions, visualize_graph, latent_edge_hist


class TestVisualization(unittest.TestCase):
    def test_get_node_positions(self):
        env = grid2op.make("l2rpn_case14_sandbox")
        obs_space = BusConnectionsGraphObsSpace
        node_positions = get_node_positions(env, obs_space)
        print(node_positions.shape)
        self.assertEqual(node_positions.shape[1], 2)
        self.assertEqual(len(node_positions.shape), 2)
        x = node_positions[:, 0]
        y = node_positions[:, 1]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c='blue', s=20)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Node Positions")
        plt.axis('equal')  # keeps aspect ratio square
        plt.grid(True)
        # plt.show()
        plt.close()

    def test_visualize_latent_graph(self):
        env = grid2op.make("l2rpn_case14_sandbox")
        obs_space = BusConnectionsGraphObsSpace(env.observation_space)
        obs = env.reset()

        num_edge_types = 3
        num_edges_fully_connected = obs_space.num_nodes * (obs_space.num_nodes - 1)

        edge_type_probs = torch.rand((num_edges_fully_connected, num_edge_types)) + 0.0001
        edge_type_probs /= edge_type_probs.sum(dim=-1, keepdim=True)
        fig = visualize_graph(obs_space.num_nodes, edge_index=obs_space.to_gym(obs)[EDGE_INDEX],
                              latent_edge_probs=edge_type_probs, skip_first_edge_type=True,
                              node_positions=get_node_positions(env, obs_space.__class__))
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


