from typing import Optional

import torch
from torch import nn, Tensor

from common import MLP
from nri.utils import Node2Edge, Edge2Node, fully_connected_edge_index_per_batch, edge_membership_mask


class Encoder(nn.Module):
    """
    Encoder closely inspired by Kipf et al.
    Predicts posterior p(z|x) where x encodes the current observation and z is a distribution over edge types
    for each edge of the fully meshed graph.
    """
    def __init__(self, x_dim: int, hidden_dim: int, num_edge_types: int = 2, dropout_prob=0.):
        """
        Constructor
        :param x_dim: number of input features for nodes
        :param hidden_dim: the dimensionality of hidden layers
        :param num_edge_types: the number of possible types per edge (defaults to 2 (edge doesn't exist or edge exists))
        :param dropout_prob: dropout probability (defaults to 0.)
        """
        super(Encoder, self).__init__()
        # learns latent representation of node features
        self.f_emb = MLP(
            input_features=x_dim,
            hidden_dim=hidden_dim,
            output_features=hidden_dim,
            dropout_prob=dropout_prob,
            do_batch_norm=False,
        )
        # maps 2 node embeddings to one edge embedding
        self.node2edge_1 = Node2Edge(
            x_dim=hidden_dim,
            hidden_dim=hidden_dim,
            e_dim=hidden_dim,
            dropout_prob=dropout_prob
        )
        # update node values depending on adjacent edge values
        self.edge2node = Edge2Node(
            e_dim=hidden_dim + 1,
            hidden_dim=hidden_dim,
            x_dim=hidden_dim,
            dropout_prob=dropout_prob
        )
        # maps 2 node embeddings and previous edge embedding to new edge embedding
        self.node2edge_2 = Node2Edge(
            x_dim=hidden_dim,
            hidden_dim=hidden_dim,
            e_dim=hidden_dim,
            dropout_prob=dropout_prob
        )
        # maps to output dimensions
        self.fc_out = MLP(
            input_features=hidden_dim * 2 + 1,
            hidden_dim=hidden_dim,
            output_features=num_edge_types,
            dropout_prob=dropout_prob,
            do_batch_norm=False,
        )

    def forward(self, x: Tensor, powerline_edge_index: Optional[Tensor] = None, edge_set: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> Tensor:
        """
        Predicts edge type for each edge in edge_index.
        :param x: node features [N, X_dim]
        :param powerline_edge_index: Optional edge index indicating existing edges [2, E']. Existence information will be appended on edge embeddings.
        :param edge_set: node adjacency [2, E]. Only latent edges that are included in this argument are detected. Per default this is fully meshed.
        :param batch: indicates which batch each node belongs to [N]
        :return: edge type prediction [E, num_edge_types]
        """
        N, x_dim = x.shape[-2:]
        batch = batch if batch is not None else torch.zeros(N).to(x.device)
        edge_set = edge_set if edge_set is not None else fully_connected_edge_index_per_batch(batch, x.device)
        _, E = edge_set.shape
        if powerline_edge_index is not None:
            mask = edge_membership_mask(edge_set, powerline_edge_index).to(device=x.device, dtype=x.dtype).unsqueeze(-1)  # [E]
        else:
            mask = torch.zeros((E,)).to(device=x.device, dtype=x.dtype).unsqueeze(-1)  # [E]

        # embed each node in lower dimensional space
        x = self.f_emb(x)  # 2-layer ELU net per node

        # v -> e
        e = self.node2edge_1.forward(x, edge_set)
        e_skip = e

        # e -> v
        e = torch.cat([e, mask], dim=-1)  # [E, hidden + 1]
        x = self.edge2node.forward(e, edge_set)

        # v -> e
        e = self.node2edge_2.forward(x, edge_set)

        e = torch.cat([e, e_skip, mask], dim=-1)
        return self.fc_out(e)
