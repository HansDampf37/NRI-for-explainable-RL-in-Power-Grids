from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, LongTensor
from torch_geometric.utils import to_dense_batch


class CentralityEncoding(nn.Module):
    def __init__(self, max_degree: int, node_dim: int):
        """
        :param max_degree: max degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.node_dim = node_dim
        self.max_degree = max_degree
        self.z = nn.Parameter(torch.randn((max_degree, node_dim)))

    def forward(self, degree: torch.LongTensor) -> torch.Tensor:
        """
        :param degree: degree of each node
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        return self.z[torch.clamp(degree, min=0, max=self.max_degree - 1)]


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance

        self.b = nn.Parameter(torch.randn(self.max_path_distance + 1))

    def forward(self, node_paths_length: torch.Tensor) -> torch.Tensor:
        """
        :param node_paths_length: tensor of shape [num_nodes, num_nodes] containing path lengths
        :return: torch.Tensor, spatial Encoding matrix
        """
        mask_of_nonexistent_paths = (node_paths_length == 0)
        indices = torch.clamp(node_paths_length-1, min=0, max=self.max_path_distance)
        spatial_matrix = F.embedding(indices, self.b.unsqueeze(1)).squeeze(-1)
        spatial_matrix[mask_of_nonexistent_paths] = 0

        return spatial_matrix


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_qk: int, dim_v: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_qk: key/query node feature matrix input number dimension
        :param dim_v: value node feature matrix input number of dimension
        """
        super().__init__()
        self.dim_qk = dim_qk

        self.q = nn.Linear(dim_in, dim_qk)
        self.k = nn.Linear(dim_in, dim_qk)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self,
                x: torch.Tensor,
                b: torch.Tensor,
                batch: Optional[torch.LongTensor]=None,
                return_attn_logits: bool = False) -> torch.Tensor:
        """
        :param x: node feature matrix [BxN, x_dim],
        :param b: spatial encoding in the shape [B, N, N]
        :param batch: pointer tensor for batching [N, ]
        :param return_attn_logits: whether to return attention-logits (before softmax) instead of the scaled values
        :return: torch.Tensor, node embeddings after attention operation
        """
        # use separate batch dimension
        BxN = x.shape[-2]
        batch = batch if batch is not None else torch.zeros(BxN, dtype=torch.long, device=x.device)
        x_dense, mask = to_dense_batch(x, batch)
        B, N, _ = x_dense.shape
        assert torch.all(mask), "Different number of nodes across batches is not allowed."

        query = self.q(x_dense) # [B, N, dim_qk]
        key = self.k(x_dense) # [B, N, dim_qk]

        a = self.compute_a(key, query) + b # [B, N, N]

        if return_attn_logits:
            return a

        value = self.v(x_dense) # [B, N, dim_v]
        softmax = torch.softmax(a, dim=-1) # [B, N, N]
        x = torch.matmul(softmax, value) # [B, N, dim_v]

        x = x.view(B * N, -1)
        return x

    def compute_a(self, key: torch.Tensor, query: torch.Tensor):
        """
        Computes attention weights between query and key.

        :param key: the key [B, N, dim_qk]
        :param query: the query [B, N, dim_qk]
        :return: attention weights [B, N, N]
        """
        return torch.matmul(query, key.transpose(1, 2)) / self.dim_qk ** 0.5


# FIX: PyG attention instead of regular attention, due to specificity of GNNs
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_qk: int, dim_v: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_qk: query/key node feature matrix input number dimension
        :param dim_v: value node feature matrix input number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList([GraphormerAttentionHead(dim_in, dim_qk, dim_v) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self,
                x: torch.Tensor,
                b: torch.Tensor,
                batch: Optional[LongTensor]=None) -> torch.Tensor:
        """
        :param x: node feature matrix [BxN, x_dim]
        :param b: spatial Encoding matrix [B, N, N]
        :param batch: batch pointer that shows graph indexes in batch of graphs [BxN,]
        :return: torch.Tensor, node embeddings after all attention heads [BxN, x_dim]
        """
        return self.linear(torch.cat([attention_head(x, b, batch) for attention_head in self.heads], dim=-1))


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, n_heads, ff_dim):
        """
        :param node_dim: node feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_v=node_dim,
            dim_qk=node_dim,
            num_heads=n_heads,
        )
        self.ln_1 = nn.LayerNorm(self.node_dim)
        self.ln_2 = nn.LayerNorm(self.node_dim)
        self.ff = nn.Sequential(
                    nn.Linear(self.node_dim, self.ff_dim),
                    nn.GELU(),
                    nn.Linear(self.ff_dim, self.node_dim)
        )


    def forward(self,
                x: torch.Tensor,
                b: torch.Tensor,
                batch: Optional[LongTensor] = None) -> torch.Tensor:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param b: spatial Encoding matrix
        :param batch: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), b, batch) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new
