import unittest

import grid2op
import torch
from matplotlib import pyplot as plt

from common.graph_structured_observation_space import BusConnectionsGraphObsSpace, EDGE_INDEX
from visualization.utils import get_node_positions, visualize_powergrid


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
        plt.show()

    def test_visualize_latent_graph(self):
        env = grid2op.make("l2rpn_case14_sandbox")
        obs_space = BusConnectionsGraphObsSpace(env.observation_space)
        obs = env.reset()
        edge_index = obs_space.to_gym(obs)[EDGE_INDEX]
        num_edges = 10
        num_types = 4
        predicted_edge_index = torch.randint(low=0, high=obs_space.num_node, size=(2, num_edges))
        predicted_edge_types = torch.randint(low=0, high=num_types, size=(1, num_edges))
        typed_edge_index = torch.concatenate([predicted_edge_index, predicted_edge_types], dim=0)
        fig = visualize_powergrid(typed_edge_index, edge_index, True, get_node_positions(env, obs_space.__class__))
        plt.show()
        plt.close(fig)


