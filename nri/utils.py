from typing import Union
from common.MLP import MLP
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable



def fully_connected_edge_index(num_nodes: int, self_loops: bool = False) -> Tensor:
    """
    Create an edge index representing a fully connected graph with num_nodes nodes.
    :param num_nodes: Number of nodes in the graph.
    :param self_loops: If true, create self-loops.
    """
    senders, receivers = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    edge_index = torch.stack([senders.flatten(), receivers.flatten()], dim=0)
    if not self_loops:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index

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
    def __init__(self, edge_dim: int, hidden_dim: int, node_dim: int, dropout_prob=0.):
        """
        Constructor.
        :param edge_dim: edge features in the input
        :param hidden_dim: hidden dimension for the MLP
        :param node_dim: node features in the output of this module
        :param dropout_prob: dropout probability
        """
        # maps adjacent edge embeddings onto a new node embedding
        super(Edge2Node, self).__init__()
        self.phi = MLP(
            input_features=edge_dim,
            hidden_dim=hidden_dim,
            output_features=node_dim,
            dropout_prob=dropout_prob
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

class EdgeNode2Node(nn.Module):
    """
    Implements the edge to node message passing by Kipf et al. in their decoder.
    In contrast to the simple Edge2Node module this module inputs previous node features in the mlp
    """
    def __init__(self, node_in_dim: int, edge_dim: int, hidden_dim: int, node_out_dim: int, dropout_prob=0.):
        """
        Constructor.
        :param node_in_dim: node features in the input
        :param edge_dim: edge features in the input
        :param hidden_dim: hidden dimension for the MLP
        :param node_out_dim: node features in the output of this module
        :param dropout_prob: dropout probability
        """
        # maps adjacent edge embeddings + node embeddings onto a new node embedding
        super(EdgeNode2Node, self).__init__()
        self.phi = MLP(
            input_features=edge_dim + node_in_dim,
            hidden_dim=hidden_dim,
            output_features=node_out_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, x: Tensor, e: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates node + edge features into node features.
        :param x: node features [B, N, X_dim]
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
        return self.phi(torch.cat([agg, x], dim=-1))
