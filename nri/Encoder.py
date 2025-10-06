from typing import Optional

import torch
from torch import nn, Tensor

from common.MLP import MLP
from nri.utils import fully_connected_edge_index, Node2Edge, Edge2Node


class Encoder(nn.Module):
    """
    Encoder closely inspired by Kipf et al.
    Predicts posterior p(z|x) where x encodes several timesteps of our environment and z is a distribution over edge types
    for each edge of the fully meshed graph.
    """
    def __init__(self, x_dim: int, trajectory_length: int, hidden_dim: int, e_out_dim: int = 2, do_prob=0.):
        """
        Constructor
        :param x_dim: number of input features for nodes
        :param trajectory_length: the number of timesteps in the trajectories
        :param hidden_dim: the dimension of hidden layers
        :param e_out_dim: the number of possible types per edge (defaults to 2 (edge doesn't exist or edge exists))
        :param do_prob: dropout probability (defaults to 0.)
        """
        super(Encoder, self).__init__()
        # learns latent smaller representation of node features
        self.f_emb = MLP(
            input_features=x_dim * trajectory_length,
            hidden_dim=hidden_dim,
            output_features=hidden_dim,
            dropout_prob=do_prob
        )
        # maps 2 node embeddings to one edge embedding
        self.node2edge_1 = Node2Edge(
            x_dim=hidden_dim,
            hidden_dim=hidden_dim,
            e_dim=hidden_dim,
            dropout_prob=do_prob
        )
        # update node values depending on adjacent edge values
        self.edge2node = Edge2Node(
            e_dim=hidden_dim,
            hidden_dim=hidden_dim,
            x_dim=hidden_dim,
            dropout_prob=do_prob
        )
        # maps 2 node embeddings and previous edge embedding to new edge embedding
        self.node2edge_2 = Node2Edge(
            x_dim=hidden_dim,
            hidden_dim=hidden_dim,
            e_dim=hidden_dim,
            dropout_prob=do_prob
        )
        # maps to output dimensions
        self.fc_out = MLP(
            input_features=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_features=e_out_dim,
            dropout_prob=do_prob
        )

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """
        Predicts edge type for each edge in edge_index.
        :param x: node features [B, T, N, X_dim]
        :param edge_index: node adjacency [2, E]. Only latent edges that are included in this argument are detected. Per default this is fully meshed.
        """
        B, T, N, x_dim = x.shape
        x = x.view(B, N, T * x_dim)
        edge_index = edge_index if edge_index is not None else fully_connected_edge_index(N)

        # embed each node in lower dimensional space
        x = self.f_emb(x)  # 2-layer ELU net per node

        # v -> e
        e = self.node2edge_1(x, edge_index)
        e_skip = e

        # e -> v
        x = self.edge2node(e, edge_index)

        # v -> e
        e = self.node2edge_2(x, edge_index)
        e = torch.cat([e, e_skip], dim=-1)

        return self.fc_out(e)
