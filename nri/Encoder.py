from typing import Union, Optional

import numpy as np
import torch
from torch import nn, Tensor

from common.MLP import MLP
from nri.utils import fully_connected_edge_index


class Encoder(nn.Module):
    """
    Encoder closely inspired Kipf et el.
    Predicts posterior p(z|x) where x is one or more time steps of our environment and z is a distribution over edge types
    for each edge of the fully meshed graph.
    """
    def __init__(self, x_dim: int, hidden_dim: int, out_e_dim: int = 2, do_prob=0.):
        """
        Constructor
        :param x_dim: number of input features in nodes * nb_timesteps
        :param hidden_dim: the dimension of hidden layers
        :param out_e_dim: the number of possible types per edge (defaults to 2 (doesn't exist or edge exists))
        :param do_prob: dropout probability (defaults to 0.)
        """
        super(Encoder, self).__init__()
        # learns latent smaller representation of node features
        self.f_emb = MLP(
            input_features=x_dim,
            hidden_dim=hidden_dim,
            output_features=hidden_dim,
            dropout_prob=do_prob
        )
        # maps 2 node embeddings to one edge embedding
        self.node2edge_1 = Node2Edge(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            edge_dim=hidden_dim,
            dropout_prob=do_prob
        )
        # update node values depending on adjacent edge values
        self.edge2node = Edge2Node(
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            node_dim=hidden_dim,
            do_prob=do_prob
        )
        # maps 2 node embeddings and previous edge embedding to new edge embedding
        self.node2edge_2 = Node2Edge(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            edge_dim=hidden_dim,
            dropout_prob=do_prob
        )
        # maps to output dimensions
        self.fc_out = MLP(
            input_features=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_features=out_e_dim,
            dropout_prob=do_prob
        )

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """
        Predicts edge type for each edge in edge_index.
        :param x: node features [B, N, nb_timesteps * X_dim]
        :param edge_index: node adjacency [2, E]. Only latent edges that are included in this argument are detected. Per default this is fully meshed.
        """
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        num_trajectory = x.size(0)
        num_nodes = x.size(1)
        trajectory_len = x.size(2)
        node_features = x.size(3)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = x.view(num_trajectory, num_nodes, trajectory_len * node_features)
        edge_index = edge_index or fully_connected_edge_index(num_nodes=num_nodes, self_loops=False).to(x.device)

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


class Node2Edge(nn.Module):
    """
   Implements the Node to edge message passing by Kipf et al.
   """
    def __init__(self, node_dim: int, hidden_dim: int, edge_dim: int, dropout_prob=0.):
        # maps 2 node embeddings to one edge embedding
        super(Node2Edge, self).__init__()
        self.psi = MLP(
            input_features=node_dim * 2,
            hidden_dim=hidden_dim,
            output_features=edge_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, x: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates node features into edge features.
        :param x: node features [B, N, X_dim]
        :param edge_index: adjacency information [2, E]
        """
        senders = edge_index[0]  # j
        receivers = edge_index[1]  # i
        # gather x_j and x_i per edge
        x_j = x[:, senders, :]  # [B, E, x_dim]
        x_i = x[:, receivers, :]  # [B, E, x_dim]
        node_aggr = torch.cat([x_i, x_j], dim=-1)
        return self.psi(node_aggr)

class Edge2Node(nn.Module):
    """
    Implements the edge to node message passing by Kipf et al. in their encoder
    """
    def __init__(self, edge_dim: int, hidden_dim: int, node_dim: int, do_prob=0.):
        # maps adjacent edge embeddings onto a new node embedding
        super(Edge2Node, self).__init__()
        self.phi = MLP(
            input_features=edge_dim,
            hidden_dim=hidden_dim,
            output_features=node_dim,
            dropout_prob=do_prob
        )

    def forward(self, e: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates edge features into node features.
        :param e: edge features [B, E, E_dim]
        :param edge_index: adjacency information [2, E]
        """
        receivers = edge_index[1]
        # aggregate edge messages into nodes by receiver index
        with torch.no_grad():
            batch_size = e.size(0)  # B
            num_nodes = int(edge_index.max().item()) + 1
            agg = e.new_zeros((batch_size, num_nodes, e.size()[-1]))
        # index_add_ to sum messages into receivers rows
        agg.index_reduce_(dim=1, index=receivers, source=e, reduce="mean")
        return self.phi(agg)