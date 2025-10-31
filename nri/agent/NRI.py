from typing import Optional

import torch
import torch.nn.functional as f
from torch import nn, Tensor

from nri.Sampling import GumbelSoftmax
from nri.utils import fully_connected_edge_index_per_batch
from nri.nri_rl.Encoder import Encoder
from nri.nri_rl.GNN import NRIInformedGNN


class NRI_GNN(nn.Module):
    """
    Combines an NRI encoder with an NRI-informed GN.

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
        self.gnn: NRIInformedGNN = NRIInformedGNN(
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            x_out_dim=x_out_dim,
            n_layers=2,
            num_edge_types=num_edge_types,
            dropout_prob=dropout_prob,
            residual=True,
            skip_last=True,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x (Tensor): Node features of shape [N, x_dim].
            edge_index (Tensor): Graph connectivity in COO format [2, E]. (defaults to fully connected per batch)
            batch (Tensor): Batch vector mapping each node to its graph [N]. (defaults to every node in the same batch)

        Returns:
            tuple[Tensor, Tensor]:
                predictions (Tensor): Graph-level features [B, x_out_dim].
                p_z_given_x (Tensor): Soft edge-type posterior probabilities [E, K].
        """
        N, x_dim = x.shape[-2:]
        batch = batch if batch is not None else torch.zeros(N).to(x.device)
        edge_index = edge_index if edge_index is not None else fully_connected_edge_index_per_batch(batch, x.device)

        encoder_logits: Tensor = self.encoder.forward(x=x, edge_index=edge_index, batch=batch)
        p_z_given_x: Tensor = f.softmax(encoder_logits, dim=-1)
        p_one_hot: Tensor = self.gumbel_softmax.forward(x=encoder_logits)
        predictions: Tensor = self.gnn.forward(x=x, edge_index=edge_index, edge_type_posterior=p_one_hot, batch=batch)
        return predictions, p_z_given_x
