import torch
from torch import nn, Tensor
from torch_geometric.nn import global_mean_pool, GCNConv, BatchNorm

from common.MLP import MLP


class RAGNN(nn.Module):
    """
    The Relation Aware GNN (RAGNN) performs message passing conditioned on edge-type probabilities
    predicted by a Neural Relational Inference (NRI) encoder.

    The NRI encoder output represents a distribution over K possible edge types for
    every edge, shaped [E, K]. In the simplest case, K=2 corresponds to "edge exists"
    and "no edge".

    Within a layer, for each edge type k, the model applies a distinct GCNConv layer using the corresponding
    edge weights edge_type_posterior[:, k]. The outputs from all edge types are summed, followed by a
    non-linearity and dropout.

    The overall pipeline:
    1. Project input node features x ∈ [N, X_in] into a hidden space.
    2. Apply relations aware message passing layers (each conditioned on edge-type weights).
    3. Project hidden representations into the output feature space.
    4. Aggregate node features into graph-level embeddings via global mean pooling.

    Input shapes:
        x: [N, X_in]
        edge_index: [2, E]
        edge_type_posterior: [E, K]
        batch: [N]  # graph membership of each node

    Output shape:
        [B, X_out]
    """


    def __init__(
            self,
            x_dim: int,
            hidden_dim: int,
            x_out_dim: int,
            num_layers: int = 3,
            num_edge_types: int = 2,
            skip_last: bool = True,
            dropout_prob: float = 0.0,
            residual: bool = True,
    ):
        """
        Instantiate relations aware GNN.

        :param x_dim: input node feature dimension
        :param hidden_dim: hidden dim for node embeddings
        :param x_out_dim: output node feature dimension
        :param num_layers: number of message passing layers (default: 3)
        :param num_edge_types: number of edge types (K) (default: 2)
        :param skip_last: whether to skip the last edge type (since it encodes no existing) (default: True)
        :param dropout_prob: dropout probability (default 0)
        :param residual: whether to use residual connections (default: True)
        """
        super().__init__()
        self.n_layers = num_layers
        self.n_edge_types = num_edge_types
        self.residual = residual
        # precompute the edge type range
        self.edge_type_range = range(self.n_edge_types - 1) if skip_last else range(self.n_edge_types)

        # initial projection to working dims
        self.node_proj = MLP(
            input_features=x_dim,
            output_features=hidden_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            do_batch_norm=False
        )
        self.bn_node_proj = BatchNorm(hidden_dim)

        # build message passing layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    improved=True,
                    add_self_loops=True,
                ) for _ in self.edge_type_range
            ]) for _ in range(num_layers)
        ])
        # batch norm activation and dropout in every hidden layer
        self.bn_message_passing = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.activation_function = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob)

        self.final = MLP(
            input_features=hidden_dim,
            output_features=x_out_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            do_batch_norm=False
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_type_posterior: Tensor, batch: Tensor) -> Tensor:
        """
        Forward pass. Accepts only graphs batched via batch vector.

        :param x: node features [N, x_in_dim]
        :param edge_index: models adjacency [2, E]
        :param batch: indicates which batch each node belongs to [N]
        :param edge_type_posterior: edge type probabilities (normally computed by NRI encoder) [E, K]
        :return: output features [B, x_out_dim]
        """
        # embed input x in hidden space
        assert edge_type_posterior.size(0) == edge_index.size(1)
        x_h = self.node_proj(x) # [N, x_hidden_dim]
        x_h = self.bn_node_proj(x_h)

        # push through message passing
        for l, mp in enumerate(self.layers):
            outs = [
                mp[edge_type](x=x_h, edge_index=edge_index, edge_weight=edge_type_posterior[:, edge_type])
                for edge_type in self.edge_type_range
            ]
            x_h_ks = torch.stack(outs).sum(0)
            x_h_ks = self.bn_message_passing[l](x_h_ks)
            x_h_ks = self.activation_function(x_h_ks)
            x_h = x_h + x_h_ks if self.residual else x_h_ks
            x_h = self.dropout(x_h)

        # project hidden space onto output space
        x_final = self.final(x_h)  # [N, x_out_dim]

        node_pool = global_mean_pool(x_final, batch)
        return node_pool

class BaselineGNN(nn.Module):
    """
    The Baseline-GNN performs message passing on powerline edges and acts as a baseline to the RA_GNN.
    Similar to the RA_GNN, this model stacks multiple message passing layers. However, each layer only uses one GCNConv
    layer, ignoring different edge-types.

    Input shapes:
        x: [N, X_in]
        edge_index: [2, E]
        batch: [N]  # graph membership of each node

    Output shape:
        [B, X_out]
    """


    def __init__(
            self,
            x_dim: int,
            hidden_dim: int,
            x_out_dim: int,
            num_layers: int = 3,
            dropout_prob: float = 0.0,
            residual: bool = True,
    ):
        """
        Instantiate relations aware GNN.

        :param x_dim: input node feature dimension
        :param hidden_dim: hidden dim for node embeddings
        :param x_out_dim: output node feature dimension
        :param num_layers: number of message passing layers (default: 3)
        :param dropout_prob: dropout probability (default 0)
        :param residual: whether to use residual connections (default: True)
        """
        super().__init__()
        self.n_layers = num_layers
        self.residual = residual
        # precompute the edge type range

        # initial projection to working dims
        self.node_proj = MLP(
            input_features=x_dim,
            output_features=hidden_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            do_batch_norm=False
        )
        self.bn_node_proj = BatchNorm(hidden_dim)

        # build message passing layers
        self.layers = nn.ModuleList([
            GCNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                improved=True,
                add_self_loops=True
            ) for _ in range(num_layers)
        ])
        # batch norm activation and dropout in every hidden layer
        self.bn_message_passing = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.activation_function = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob)

        self.final = MLP(
            input_features=hidden_dim,
            output_features=x_out_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            do_batch_norm=False
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """
        Forward pass. Accepts only graphs batched via batch vector.

        :param x: node features [N, x_in_dim]
        :param edge_index: models adjacency [2, E]
        :param batch: indicates which batch each node belongs to [N]
        :return: output features [B, x_out_dim]
        """
        # embed input x in hidden space
        x_h = self.node_proj(x) # [N, x_hidden_dim]
        x_h = self.bn_node_proj(x_h)

        # push through message passing
        x_residual = x_h
        for l, _ in enumerate(self.layers):
            x_h = self.layers[l](x=x_h, edge_index=edge_index)
            x_h = self.bn_message_passing[l](x_h)
            x_h = self.activation_function(x_h)
            x_h = x_h + x_residual if self.residual else x_h
            x_h = self.dropout(x_h)
            x_residual = x_h

        # project hidden space onto output space
        x_final = self.final(x_h)  # [N, x_out_dim]

        node_pool = global_mean_pool(x_final, batch)
        return node_pool