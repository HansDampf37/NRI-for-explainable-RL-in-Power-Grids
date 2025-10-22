import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn, Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool

from .MLP import MLP
from .graph_structured_observation_space import EDGE_INDEX, EDGES, NODES, EDGE_MASK, GNNObservationSpace


class NodeEdgeMP(MessagePassing):
    """
    PyTorch Geometric style message passing:
      e_{j,i} = psi(e_{j,i}, x_i, x_j)
      x_i = x_i + node_update(x_i, sum_j phi(x_i, x_j, e_{j,i}))
    """

    def __init__(
        self,
        x_dim: int,
        e_dim: int,
        x_out_dim: int = None,
        e_out_dim: int = None,
        hidden_dim: int = 64,
        residual: bool = False,
        dropout_prob: float = 0.0,
    ):
        """
        Instantiate a message passing layer.

        :param x_dim: The number of node features in the input
        :param e_dim: The number of edge features in the input
        :param x_out_dim: The number of node features in the output (defaults to x_dim)
        :param e_out_dim: The number of edge features in the output (defaults to e_dim)
        :param hidden_dim: The number of hidden units in the message passing MLPs (default: 128)
        :param residual: If True, add x and e to x_out and e_out respectively (default: False). Only works of x_dim = x_out_dim and e_dim = e_out_dim
        :param dropout_prob: the probability to do dropout in the message passing MLPs (default 0)
        """
        super().__init__(aggr="add")
        self.x_dim = x_dim
        self.e_dim = e_dim
        self.x_out_dim = x_out_dim or x_dim
        self.e_out_dim = e_out_dim or e_dim
        self.residual = residual
        if residual and (x_dim != self.x_out_dim or e_dim != self.e_out_dim):
            raise ValueError("Cannot use residual when input/output dims differ.")

        # psi: update edge embedding from (e, x_i, x_j)
        self.psi = MLP(
            input_features=self.e_dim + self.x_dim * 2,
            output_features=self.e_out_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob
        )

        # message MLP: phi(x_i, x_j, e_ji)
        self.phi = MLP(
            input_features=self.x_dim * 2 + self.e_out_dim,
            output_features=self.x_out_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob
        )

        # node update MLP
        self.node_update = MLP(
            input_features=self.x_dim + self.x_out_dim,
            output_features=self.x_out_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        :param x: node features [N, node_in_dim]
        :param e: edge features [E, edge_in_dim]
        :param edge_index: models adjacency [2, E]
        :return: node features [N, node_out_dim] and edge features [E, edge_out_dim]
        """
        # update edges first
        src, target = edge_index
        x_i, x_j = x[target], x[src]
        psi_in = torch.cat([e, x_i, x_j], dim=-1)
        e_out = self.psi(psi_in)

        # message passing
        x_messages = self.propagate(edge_index=edge_index, x=x, e=e_out)
        node_in = torch.cat([x, x_messages], dim=-1)
        x_out = self.node_update(node_in)

        if self.residual:
            x_out = x + x_out
            e_out = e + e_out

        return x_out, e_out

    def message(self, x_j: Tensor, x_i: Tensor, e: Tensor) -> Tensor:
        """
        Compute messages for each edge
        :param x_j: sender features [E, x_dim]
        :param x_i: receiver features [E, x_dim]
        :param e: updated edge features [E, e_out_dim]
        :return: messages [E, x_out_dim]
        """
        phi_in = torch.cat([x_i, x_j, e], dim=-1)
        return self.phi(phi_in)


class GNNFeatureExtractor(nn.Module):
    """
    Stacks multiple MessagePassing layers. Followed by mean pooling and concatenation of pooled edge and node features.
    The output will have a dimension of out_x_dim + out_e_dim.
    """

    def __init__(
            self,
            x_dim: int,
            e_dim: int,
            x_hidden_dim: int,
            e_hidden_dim: int,
            x_out_dim: int,
            e_out_dim: int,
            n_layers: int = 3,
            dropout_prob: float = 0.0,
            residual=True
    ):
        """
        Instantiate GNN policy

        :param x_dim: input node feature dimension
        :param e_dim: input edge feature dimension
        :param x_hidden_dim: hidden dim for node embeddings
        :param e_hidden_dim: hidden dim for edge embeddings
        :param x_out_dim: output node feature dimension
        :param e_out_dim: output edge feature dimension
        :param n_layers: number of message passing layers (default: 3)
        :param dropout_prob: dropout probability (default 0)
        :param residual: do residual connections in Conv layers (default True)
        """
        super().__init__()
        self.n_layers = n_layers

        # initial projection to working dims
        self.node_proj = MLP(input_features=x_dim, output_features=x_hidden_dim, hidden_dim=x_hidden_dim)
        self.edge_proj = MLP(input_features=e_dim, output_features=e_hidden_dim, hidden_dim=e_hidden_dim)

        # build message passing layers
        self.layers = nn.ModuleList([
            NodeEdgeMP(
                x_dim=x_hidden_dim,
                e_dim=e_hidden_dim,
                x_out_dim=x_hidden_dim,
                e_out_dim=e_hidden_dim,
                hidden_dim=max(x_hidden_dim, e_hidden_dim),
                dropout_prob=dropout_prob,
                residual=residual,
            ) for _ in range(n_layers - 1)
        ])

        self.final = NodeEdgeMP(
            x_dim=x_hidden_dim,
            e_dim=e_hidden_dim,
            x_out_dim=x_out_dim,
            e_out_dim=e_out_dim,
            hidden_dim=max(x_hidden_dim, x_out_dim),
            dropout_prob=dropout_prob,
            residual=False,
        )

    def forward(self, x: Tensor, e: Tensor, edge_index: np.ndarray | Tensor, batch: np.ndarray | Tensor) -> Tensor:
        """
        Forward pass.

        :param x: node features [N, node_in_dim]
        :param e: edge features [E, edge_in_dim]
        :param edge_index: models adjacency [2, E]
        :param batch: indicates which batch each node belongs to [N]
        :return: output features [node_out_dim + edge_out_dim]
        """
        x_h = self.node_proj(x)
        e_h = self.edge_proj(e)

        for mp in self.layers:
            x_h, e_h = mp(x=x_h, e=e_h, edge_index=edge_index)

        x_final, e_final = self.final(x_h, e_h, edge_index) # [N, node_out_dim], [E, edge_out_dim]

        node_pool = global_mean_pool(x_final, batch)
        edge_pool = global_mean_pool(e_final, batch[edge_index[0]])
        return torch.cat([node_pool, edge_pool], dim=-1)


class SB3GNNWrapper(BaseFeaturesExtractor):
    """
    Wraps the GNN to accept input of gym-like dict observations. These observations are then transformed to torch
    geometric batches before being passed to the model.
    """
    def __init__(
            self,
            observation_space: GNNObservationSpace,
            x_hidden_dim: int,
            e_hidden_dim: int,
            x_out_dim: int,
            e_out_dim: int,
            n_layers: int = 3,
            dropout_prob: float = 0.0,
            residual=True
    ):
        BaseFeaturesExtractor.__init__(self, observation_space, features_dim=x_out_dim + e_out_dim)
        self.gnn_feature_extractor = GNNFeatureExtractor(
            x_dim=observation_space.x_dim,
            e_dim=observation_space.e_dim if observation_space.e_dim is not None else 0,
            x_hidden_dim=x_hidden_dim,
            e_hidden_dim=e_hidden_dim,
            x_out_dim=x_out_dim,
            e_out_dim=e_out_dim,
            n_layers=n_layers,
            dropout_prob=dropout_prob,
            residual=residual
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        node_features_batch = observations[NODES]  # [B, N, node_in_dim]
        edge_features_batch = observations.get(EDGES)  # [B, E, edge_in_dim] or None
        edge_masks_batch = observations[EDGE_MASK].bool()  # [B, E]
        edge_index_batch = observations[EDGE_INDEX].long()  # [B, 2, E]

        data_list = []
        batch_size = node_features_batch.size(0)

        for b in range(batch_size):
            node_features = node_features_batch[b]
            edge_features = edge_features_batch[b, edge_masks_batch[b]] if edge_features_batch is not None else None
            edge_index = edge_index_batch[b, :, edge_masks_batch[b]]
            data_list.append(Data(x=node_features, edge_index=edge_index, edge_attr=edge_features))

        batch: Batch = Batch.from_data_list(data_list)
        return self.gnn_feature_extractor(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

