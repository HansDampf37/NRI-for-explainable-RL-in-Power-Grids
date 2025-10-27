import logging
from typing import Union

import numpy as np
import torch
from torch import Tensor, nn

from common.MLP import MLP


def fully_connected_edge_index(num_nodes: int, device: str = "cpu", self_loops: bool = False) -> Tensor:
    """
    Create an edge index representing a fully connected graph with num_nodes nodes.
    :param num_nodes: Number of nodes in the graph.
    :param self_loops: If true, create self-loops.
    :param device: Device to use.
    """
    senders, receivers = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    edge_index = torch.stack([senders.flatten(), receivers.flatten()], dim=0)
    if not self_loops:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index.to(device=device)


class Node2Edge(nn.Module):
    """
   Implements the Node to edge message passing by Kipf et al.
   """

    def __init__(self, x_dim: int, hidden_dim: int, e_dim: int, dropout_prob=0.):
        # maps 2 node embeddings to one edge embedding
        super(Node2Edge, self).__init__()
        self.psi = MLP(
            input_features=x_dim * 2,
            hidden_dim=hidden_dim,
            output_features=e_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, x: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates node features into edge features.
        :param x: node features [... N, X_dim]
        :param edge_index: adjacency information [2, E]
        :return: edge features [..., E, E_dim]
        """
        senders = edge_index[0]  # j
        receivers = edge_index[1]  # i
        # gather x_j and x_i per edge
        x_j = x[..., senders, :]  # [B, T, E, x_dim]
        x_i = x[..., receivers, :]  # [B, T, E, x_dim]
        node_aggr = torch.cat([x_i, x_j], dim=-1)
        return self.psi(node_aggr)


class Edge2Node(nn.Module):
    """
    Implements the edge to node message passing by Kipf et al. in their encoder
    """

    def __init__(self, e_dim: int, hidden_dim: int, x_dim: int, dropout_prob=0.):
        """
        Constructor.
        :param e_dim: edge features in the input
        :param hidden_dim: hidden dimension for the MLP
        :param x_dim: node features in the output of this module
        :param dropout_prob: dropout probability
        """
        # maps adjacent edge embeddings onto a new node embedding
        super(Edge2Node, self).__init__()
        self.phi = MLP(
            input_features=e_dim,
            hidden_dim=hidden_dim,
            output_features=x_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, e: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates edge features into node features.

        :param e: edge features [..., E, E_dim]
        :param edge_index: adjacency information [2, E]
        :return: node features [..., N, X_dim]
        """
        receivers = edge_index[1]
        # aggregate edge messages into nodes by receiver index
        N = int(edge_index.max().item()) + 1
        target_shape = list(e.size())
        target_shape[-2] = N
        agg = e.new_zeros(tuple(target_shape))
        # index_reduce_ to average messages into receivers rows
        agg.index_reduce_(dim=-2, index=receivers, source=e, reduce="mean")
        return self.phi(agg)


class EdgeNode2Node(nn.Module):
    """
    Implements the edge to node message passing by Kipf et al. in their decoder.
    In contrast to the simple Edge2Node module here node features do not only depend on adjacent edge features but
    also on previous node features.
    """

    def __init__(self, x_dim: int, e_dim: int, hidden_dim: int, x_out_dim: int, dropout_prob=0.):
        """
        Constructor.
        :param x_dim: node features in the input
        :param e_dim: edge features in the input
        :param hidden_dim: hidden dimension for the MLP
        :param x_out_dim: node features in the output of this module
        :param dropout_prob: dropout probability
        """
        # maps adjacent edge embeddings + node embeddings onto a new node embedding
        super(EdgeNode2Node, self).__init__()
        self.phi = MLP(
            input_features=e_dim + x_dim,
            hidden_dim=hidden_dim,
            output_features=x_out_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, x: Tensor, e: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates node + edge features into node features.

        :param x: node features [..., N, X_dim]
        :param e: edge features [..., E, E_dim]
        :param edge_index: adjacency information [2, E]
        :return: node features [..., N, X_dim]
        """
        receivers = edge_index[1]
        # aggregate edge messages into nodes by receiver index
        N = x.size(-2)
        target_shape = list(e.size())
        target_shape[-2] = N
        agg = e.new_zeros(tuple(target_shape))
        # index_reduce_ to average messages into receivers rows
        agg.index_reduce_(dim=-2, index=receivers, source=e, reduce="mean")
        return self.phi(torch.cat([agg, x], dim=-1))


def warn_large_loss(logger: logging.Logger, predictions: Tensor, target: Tensor) -> Tensor:
    with torch.no_grad():
        per_feature_mse: Tensor = ((target - predictions).pow(2)).mean(dim=(0, 1, 2)).cpu().numpy()
        logger.warning(f"Large Loss: {per_feature_mse.mean()}\n"
                       f"The loss per feature is:\n{per_feature_mse}")
        return per_feature_mse