from typing import Optional

import torch
from torch import nn, Tensor, LongTensor
from torch_geometric.data import Data

from nri import fully_connected_edge_index_per_batch
from .graph_data_cacher import GraphDataRetriever
from .layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding, GraphormerAttentionHead


class GraphormerNRIEncoder(nn.Module):
    """
    Module to predict latent edges inspired by Kipf et al. and Ying et al.
    Predicts posterior p(z|x, A) where x encodes the current node features and A encodes the adjacency and z
    is a distribution over edge types for each edge of the fully meshed graph.
    """
    def __init__(self,
                 x_dim: int,
                 hidden_dim: int,
                 num_edge_types: int,
                 max_degree: int,
                 max_path_distance: int):
        """
        :param x_dim: input dimension of node features
        :param hidden_dim: hidden dimensions of node features
        :param num_edge_types: number of attention heads
        :param max_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.ff_dim = hidden_dim
        self.max_degree = max_degree
        self.max_path_distance = max_path_distance

        # retrieves and caches graph data
        self.graph_data = GraphDataRetriever()

        # project node features in embedding space
        self.node_in_lin = nn.Linear(self.x_dim, self.hidden_dim)

        # positional encoding is added on node features
        self.centrality_encoding = CentralityEncoding(
            max_degree=self.max_degree,
            node_dim=self.hidden_dim
        )

        # learnable added bias for attention depends on path lengths between nodes
        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        # one layer to have global information for every node
        self.l1 = GraphormerEncoderLayer(
            node_dim=self.hidden_dim,
            n_heads=self.num_edge_types,
            ff_dim=self.ff_dim
        )

        # per edge type spatial encodings
        self.spatial_encodings_edge_probs = nn.ModuleList([
            SpatialEncoding(max_path_distance=max_path_distance) for _ in range(num_edge_types)
        ])

        # predict attention logits for each edge prob
        self.edge_prob_layer = nn.ModuleList([
            GraphormerAttentionHead(dim_in=hidden_dim, dim_qk=hidden_dim, dim_v=hidden_dim) for _ in range(num_edge_types)
        ])


    def forward(self, x: Tensor, powerline_edge_index: Tensor, edge_set: Optional[Tensor] = None, batch: Optional[LongTensor] = None) -> Tensor:
        """
        Predicts posterior interaction type probabilities for each edge in edge_set given x and the adjacency
        information powerline edge index.

        :param x: node features [BxN, X_dim]
        :param powerline_edge_index: edge index indicating existing edges [2, E'].
        :param edge_set: node adjacency [2, E]. Only latent edges that are included in this argument are detected. Per default this is fully meshed for each graph.
        :param batch: indicates which batch each node belongs to [N, ]
        :return: edge type prediction [E, num_edge_types]
        """
        BxN, _ = x.shape
        batch = batch if batch is not None else torch.zeros(BxN, device=x.device, dtype=torch.long)
        B = batch.unique().numel()
        N = BxN // B
        edge_set = edge_set if edge_set is not None else fully_connected_edge_index_per_batch(batch, x.device)

        # Get graph attributes
        with torch.no_grad():
            graph_data = Data(x=x, edge_index=powerline_edge_index, batch=batch)
            in_deg, out_deg, path_dists = self.graph_data.get(graph_data=graph_data) # [BxN,], [BxN,], [B,N,N]
            node_deg = torch.max(in_deg, out_deg)

        x = self.node_in_lin(x)

        # get encodings
        x = x + self.centrality_encoding(node_deg)
        b = self.spatial_encoding(path_dists)

        # Apply encoder layers
        x = self.l1(x, b, batch)

        edge_probs_logits = torch.stack([
            edge_prob_layer(x, self.spatial_encodings_edge_probs[i](path_dists), batch, return_attn_logits=True)
            for i, edge_prob_layer in enumerate(self.edge_prob_layer)
        ], dim=-1)

        return edge_probs_logits[batch[edge_set[0]], edge_set[0] % N, edge_set[1] % N]
