import torch
from torch import nn, Tensor
from torch_geometric.nn import global_mean_pool, GCNConv

from common.MLP import MLP


class NRIInformedGNN(nn.Module):
    """
    The NRI-Informed GNN performs message passing conditioned on edge-type probabilities
    predicted by a Neural Relational Inference (NRI) encoder.

    The NRI encoder output represents a distribution over K possible edge types for
    every edge, shaped [E, K]. In the simplest case, K=2 corresponds to "edge exists"
    and "no edge".

    Within a layer, for each edge type k, the model applies a distinct GCNConv layer using the corresponding
    edge weights edge_type_posterior[:, k]. The outputs from all edge types are summed, followed by a
    non-linearity and dropout.

    The overall pipeline:
    1. Project input node features x ∈ [N, X_in] into a hidden space.
    2. Apply NRI-informed message passing layers (each conditioned on edge-type weights).
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
            x_hidden_dim: int,
            x_out_dim: int,
            n_layers: int = 3,
            n_edge_types: int = 2,
            skip_first: bool = True,
            dropout_prob: float = 0.0,
    ):
        """
        Instantiate GNN feature extractor.

        :param x_dim: input node feature dimension
        :param x_hidden_dim: hidden dim for node embeddings
        :param x_out_dim: output node feature dimension
        :param n_layers: number of message passing layers (default: 3)
        :param n_edge_types: number of edge types (K) (default: 2)
        :param skip_first: whether to skip the first edge type (since it encodes no existing) (default: True)
        :param dropout_prob: dropout probability (default 0)
        """
        super().__init__()
        self.n_layers = n_layers
        self.n_edge_types = n_edge_types
        self.skip_first = skip_first
        self.activation_function = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob)

        # initial projection to working dims
        self.node_proj = MLP(
            input_features=x_dim,
            output_features=x_hidden_dim,
            hidden_dim=x_hidden_dim,
            dropout_prob=dropout_prob
        )

        # build message passing layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                GCNConv(
                    in_channels=x_hidden_dim,
                    out_channels=x_hidden_dim,
                    improved=True,
                    add_self_loops=True,
                ) for _ in range(n_edge_types - 1 if skip_first else n_edge_types)
            ]) for _ in range(n_layers)
        ])

        self.final = MLP(
            input_features=x_hidden_dim,
            output_features=x_out_dim,
            hidden_dim=x_hidden_dim,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_type_posterior: Tensor, batch: Tensor) -> Tensor:
        """
        Forward pass. Accepts only graphs batched via batch vector. For batching use the batch argument.

        :param x: node features [N, x_in_dim]
        :param edge_index: models adjacency [2, E]
        :param batch: indicates which batch each node belongs to [N]
        :param edge_type_posterior: edge type probabilities (normally computed by NRI encoder) [E, K]
        :return: output features [B, x_out_dim]
        """
        _, E = edge_index.shape
        K = self.n_edge_types

        # embed input x in hidden space
        x_h = self.node_proj(x) # [N, x_hidden_dim]

        # push through message passing
        for mp in self.layers:
            start_index = 1 if self.skip_first else 0
            outs = [
                mp[edge_type](x_h, edge_index, edge_type_posterior[..., edge_type])
                for edge_type in range(start_index, K)
            ]
            x_h = self.activation_function(torch.stack(outs).sum(0))
            x_h = self.dropout(x_h)

        # project hidden space onto output space
        x_final = self.final(x_h)  # [N, x_out_dim]

        node_pool = global_mean_pool(x_final, batch)
        return node_pool
