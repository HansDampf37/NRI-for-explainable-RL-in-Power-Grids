"""
This script contains the relations aware FeatureExtractor (RAFeatureExtractor) and a relations unaware BaselineFeatureExtractor
with sb3 compatible APIs.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as f
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn, Tensor
from torch_geometric.utils import to_dense_batch

from common import GraphObservationSpace, NODES, EDGE_INDEX, EDGE_MASK
from nri.Sampling import GumbelSoftmax
from nri.agent.Encoder import Encoder
from nri.agent.RAGNN import RAGNN, BaselineGNN
from nri.utils import fully_connected_edge_index_per_batch


class RAFeatureExtractor(nn.Module):
    """
    Combines an NRI encoder with a relations aware GGN (RA-GNN).

    The encoder predicts edge-type logits for each edge. These logits are used both
    to compute soft edge-type probabilities (for monitoring) and to sample discrete
    edge-type assignments via the Gumbel–Softmax trick. The sampled edge types are
    then used by the NRIInformedGNN to perform conditioned message passing.

    Pipeline:
        1. Encode graph structure to edge-type logits.
        2. Compute posterior probabilities p(z|x) via softmax.
        3. Sample edge-type assignments using Gumbel–Softmax.
        4. Pass node features and sampled edges into the GNN.

    Args:
        x_dim (int): Input node feature dimension.
        hidden_dim (int): Hidden dimension shared by encoder and GNN.
        x_out_dim (int): Output node feature dimension.
        num_edge_types (int): Number of edge types (K).
        dropout_prob (float): Dropout probability used in both encoder and GNN.
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        x_out_dim: int,
        num_layers: int,
        num_edge_types: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.encoder: Encoder = Encoder(
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            dropout_prob=dropout_prob,
        )
        self.gumbel_softmax = GumbelSoftmax()
        self.gnn: RAGNN = RAGNN(
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            x_out_dim=x_out_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            dropout_prob=dropout_prob,
            residual=True,
            skip_last=True,
        )
        self.x_out_dim = x_out_dim

    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                edge_index: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x (Tensor): Node features of shape [B*N, x_dim].
            batch (Tensor): Batch vector mapping each node to its graph [B*N]. (defaults to every node in the same batch)
            edge_index (Tensor): Graph connectivity in COO format [2, B*E]. (defaults to fully connected per batch) THIS NEEDS TO HAVE A CONSISTENT NUMBER OF EDGES PER BATCH!

        Returns:
            tuple[Tensor, Tensor]:
                predictions (Tensor): Graph-level features [B, x_out_dim].
                p_z_given_x (Tensor): Soft edge-type posterior probabilities [B, E, K].
        """
        BxN, x_dim = x.shape[-2:]
        batch = batch if batch is not None else torch.zeros(BxN).to(x.device)
        edge_index = edge_index if edge_index is not None else fully_connected_edge_index_per_batch(batch, x.device)

        # get posterior
        encoder_logits: Tensor = self.encoder.forward(x=x, batch=batch, edge_index=edge_index) # [B*E, K]
        p_z_given_x: Tensor = f.softmax(encoder_logits, dim=-1) # [B*E, K]
        p_one_hot: Tensor = self.gumbel_softmax.forward(x=encoder_logits) # [B*E, K]

        # condition gnn on posterior and push x
        predictions: Tensor = self.gnn.forward(x=x, batch=batch, edge_index=edge_index, edge_type_posterior=p_one_hot)

        # transform posterior into batched format.
        edge_batch = batch[edge_index[0]] # edge is in the same batch as incident nodes
        batched_p_z_given_x, mask = to_dense_batch(p_z_given_x, edge_batch)
        assert torch.all(mask), "Different number of edges across batches is not allowed."

        return predictions, batched_p_z_given_x


class RAFeatureExtractorSB3(BaseFeaturesExtractor):
    """
    Wraps the RAFeatureExtractor to be compatible with the sb3 API.
    """

    def __init__(
            self,
            observation_space: GraphObservationSpace,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            num_edge_types: int,
            dropout_prob: float = 0.0,
    ):
        BaseFeaturesExtractor.__init__(self, observation_space, features_dim=out_dim)
        self.gnn_feature_extractor = RAFeatureExtractor(
            x_dim=observation_space.x_dim,
            hidden_dim=hidden_dim,
            x_out_dim=out_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            dropout_prob=dropout_prob,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_features_batch = observations[NODES]  # [B, N, node_in_dim]

        B, N, _ = node_features_batch.shape
        device = node_features_batch.device

        # Flatten nodes
        x = node_features_batch.reshape(B * N, -1)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        return self.gnn_feature_extractor(x=x, batch=batch)


class BaselineFeatureExtractorSB3(BaseFeaturesExtractor):
    """
    Wraps the BaselineGNN to be compatible with the sb3 API.
    """

    def __init__(
            self,
            observation_space: GraphObservationSpace,
            hidden_dim: int,
            out_dim: int,
            num_layers: int = 2,
            dropout_prob: float = 0.0,
    ):
        BaseFeaturesExtractor.__init__(self, observation_space, features_dim=out_dim)
        self.gnn: BaselineGNN = BaselineGNN(
            x_dim=observation_space.x_dim,
            hidden_dim=hidden_dim,
            x_out_dim=out_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            residual=True,
        )

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        node_features_batch = observations[NODES]  # [B, N, node_in_dim]
        edge_index_batch = observations[EDGE_INDEX]  # [B, 2, E_max]
        edge_mask = observations[EDGE_MASK]  # [B, E_max]

        B, N, _ = node_features_batch.shape
        device = node_features_batch.device

        # Flatten nodes
        x = node_features_batch.reshape(B * N, -1)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # Mask edges
        valid_edges = edge_mask.bool()
        edge_index_batch = edge_index_batch.permute(1, 0, 2)  # [2, B, E_max]
        edge_index_batch = edge_index_batch[:, valid_edges]  # [2, total_E]

        # Add per-graph node offsets
        offsets = (torch.arange(B, device=device) * N).repeat_interleave(valid_edges.sum(1))
        edge_index_batch += offsets.unsqueeze(0)

        return self.gnn(x=x, batch=batch, edge_index=edge_index_batch.to(dtype=torch.int32))

