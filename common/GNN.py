from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn, Tensor

from common.MLP import MLP


class MessagePassing(nn.Module):
    """
    Implements:
      e_{j,i}^{(k)} = psi^{(k)}( e_{j,i}^{(k-1)}, x_i^{(k-1)}, x_j^{(k-1)} )
      x_i^{(k)} = sigma( x_i^{(k-1)}, sum_{j in N(i)} phi^{(k)}( x_i^{(k-1)}, x_j^{(k-1)}, e_{j,i}^{(k)} ) )
    Inputs:
      x: [N, x_dim]
      e: [E, e_dim]
      edge_index: [2, E] tensor of dtype long. edge_index[0] = senders j, edge_index[1] = receivers i.
    Returns:
      x_out: [N, x_dim_out]  (by default same dim)
      e_out: [E, e_dim_out]
    """

    def __init__(
            self,
            x_dim: int,
            e_dim: int,
            output_x_dim: Optional[int] = None,
            output_e_dim: Optional[int] = None,
            hidden_dim: int = 64,
            residual: bool = False,
            dropout_prob: float = 0.0,
    ):
        """
        Instantiate a message passing layer.

        :param x_dim: The number of node features in the input
        :param e_dim: The number of edge features in the input
        :param output_x_dim: The number of node features in the output (defaults to x_dim)
        :param output_e_dim: The number of edge features in the output (defaults to e_dim)
        :param hidden_dim: The number of hidden units in the message passing MLPs (default: 128)
        :param residual: If True, add x and e to x_out and e_out respectively (default: False). Only works of x_dim = x_out_dim and e_dim = e_out_dim
        :param dropout_prob: the probability to do dropout in the message passing MLPs (default 0)
        """
        super(MessagePassing, self).__init__()
        self.x_dim = x_dim
        self.e_dim = e_dim
        self.x_out_dim = output_x_dim or x_dim
        self.e_out_dim = output_e_dim or e_dim
        self.residual = residual
        if residual and (x_dim != output_x_dim or e_dim != output_e_dim):
            raise ValueError("Cannot use residual connections. x_dim != x_out_dim or e_dim != e_out_dim.")

        # psi: update edge embedding from (e, x_i, x_j)
        psi_in = self.e_dim + self.x_dim + self.x_dim
        self.psi = MLP(
            input_features=psi_in,
            output_features=self.e_out_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob
        )

        # phi: message fn from (x_i, x_j, e_{j,i})
        phi_in = self.x_dim + self.x_dim + self.e_out_dim
        self.phi = MLP(
            input_features=phi_in,
            output_features=self.x_out_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob
        )

        # node_update: optionally process concatenated (x_i_old, aggregated_messages)
        node_update_in = self.x_dim + self.x_out_dim
        self.node_update = MLP(
            input_features=node_update_in,
            output_features=self.x_out_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        :param x: node features [B, N, node_in_dim]
        :param e: edge features [B, E, edge_in_dim]
        :param edge_index: models adjacency [2, E]
        :return: node features [B, N, node_out_dim] and edge features [B, E, edge_out_dim]
        """
        senders = edge_index[0]  # j
        receivers = edge_index[1]  # i
        # gather x_j and x_i per edge
        x_j = x[:, senders, :]  # [B, E, x_dim]
        x_i = x[:, receivers, :]  # [B, E, x_dim]

        # update edges: e_out = psi(e_old, x_i, x_j)
        psi_in = torch.cat([e, x_i, x_j], dim=-1)
        e_out = self.psi(psi_in)  # [E, e_out_dim]

        # compute messages m_{j->i} = phi(x_i, x_j, e_{j,i})
        phi_in = torch.cat([x_i, x_j, e_out], dim=-1)
        messages = self.phi(phi_in)  # [E, x_out_dim]

        # aggregate (sum) messages into nodes by receiver index
        B = x.size(0)
        N = x.size(1)
        agg = x.new_zeros((B, N, messages.size(-1)))
        # index_add_ to sum messages into receivers rows
        agg.index_add_(index=receivers, source=messages, dim=1)

        # node update: combine old x and aggregated messages
        node_in = torch.cat([x, agg], dim=-1)
        x_out = self.node_update(node_in)

        if self.residual:
            e_out = e + e_out
            x_out = x + x_out

        return x_out, e_out


class GNNFeatureExtractor(nn.Module):
    """
    Stacks multiple MessagePassing layers.
    """

    def __init__(
            self,
            x_dim: int,
            e_dim: int,
            hidden_x_dim: int,
            hidden_edge: int,
            node_out_dim: int,
            edge_out_dim: int,
            n_layers: int = 3,
            dropout_prob: float = 0.0,
            residual=True
    ):
        """
        Instantiate our GNN policy

        :param x_dim: input node feature dimension
        :param e_dim: input edge feature dimension
        :param hidden_x_dim: hidden dim for node embeddings
        :param hidden_edge: hidden dim for edge embeddings
        :param node_out_dim: output node feature dimension
        :param edge_out_dim: output edge feature dimension
        :param n_layers: number of message passing layers (default: 3)
        :param dropout_prob: dropout probability (default 0)
        :param residual: do residual connections in Conv layers (default True)
        """
        super().__init__()
        self.n_layers = n_layers

        # initial projection to working dims
        self.node_proj = MLP(input_features=x_dim, output_features=hidden_x_dim, hidden_dim=hidden_x_dim)
        self.edge_proj = MLP(input_features=e_dim, output_features=hidden_edge, hidden_dim=hidden_edge)

        # build message passing layers
        self.layers = nn.ModuleList([
            MessagePassing(
                x_dim=hidden_x_dim,
                e_dim=hidden_edge,
                output_x_dim=hidden_x_dim,
                output_e_dim=hidden_edge,
                hidden_dim=max(hidden_x_dim, hidden_edge),
                dropout_prob=dropout_prob,
                residual=residual,
            ) for _ in range(n_layers - 1)
        ])

        self.final = MessagePassing(
            x_dim=hidden_x_dim,
            e_dim=hidden_edge,
            output_x_dim=node_out_dim,
            output_e_dim=edge_out_dim,
            hidden_dim=max(hidden_x_dim, node_out_dim),
            dropout_prob=dropout_prob,
            residual=False,
        )

    def forward(self, x: Tensor, e: Tensor, edge_index: np.ndarray) -> Tensor:
        """
        Forward pass.

        :param x: node features [B, N, node_in_dim]
        :param e: edge features [B, E, edge_in_dim]
        :param edge_index: models adjacency [2, E]
        :return: output features [B, node_out_dim + edge_out_dim]
        """
        x_h = self.node_proj(x)
        e_h = self.edge_proj(e)

        for mp in self.layers:
            x_h, e_h = mp(x_h, e_h, edge_index)

        x_final, e_final = self.final(x_h, e_h, edge_index) # [B, N, node_out_dim], [B, E, edge_out_dim]

        node_pool = x_final.mean(axis=1)
        edge_pool = e_final.mean(axis=1)
        return torch.cat([node_pool, edge_pool], dim=-1)
