from typing import Optional

import torch
from torch import nn, Tensor

from nri.utils import EdgeNode2Node, Node2Edge, fully_connected_edge_index


class Decoder(nn.Module):
    """
    Decoder closely inspired by Kipf et al.
    Predicts node features for future timesteps given the history of node features as well as distributions p(z|x) over
    edge types for each edge predicted by the encoder.
    """

    def __init__(
        self,
        x_dim: int,
        num_edge_types: int,
        hidden_dim: int,
        dropout_prob: float = 0.0,
        skip_first: bool = False,
    ):
        """
        Constructor
        :param x_dim: number of input features for nodes
        :param hidden_dim: hidden dimension for MLPs used
        :param dropout_prob: dropout probability (defaults to 0.)
        :param skip_first: whether to skip the first edge type (this edge type commonly encoder the no-edge type) (defaults to False)
        """
        super().__init__()

        self.num_edge_types = num_edge_types
        self.skip_first = skip_first

        self.node2edge_list = nn.ModuleList([
            Node2Edge(
                node_dim=x_dim,
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_edge_types)
        ])

        self.edge_node2node = EdgeNode2Node(
            node_in_dim=x_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            node_out_dim=x_dim,
            dropout_prob=dropout_prob
        )

    def single_step_forward(
        self,
        x: Tensor,
        edge_types: Tensor,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform one step of message passing and node update.

        :param x: node features [B, N, X_dim]
        :param edge_types: edge type predicted by the encoder # [B, E, edge_types]
        :param edge_index: edge index used by the encoder [2, E]. Defaults to fully meshed edge index.
        :return: node features for next time step [B, N, X_dim]
        """
        start_idx = 1 if self.skip_first else 0
        edge_index = edge_index if edge_index is not None else fully_connected_edge_index(x.size(dim=1))
        edge_types = edge_types.unsqueeze(-1) # add 1-d dim for features so we can broadcast with edge-features
        edge_properties_all = []
        for k in range(start_idx, self.num_edge_types):
            edge_properties = edge_types[:, :, k, :] * self.node2edge_list[k](x, edge_index)
            edge_properties_all.append(edge_properties)
        e = sum(edge_properties_all)

        delta = self.edge_node2node(x, e, edge_index)
        return x + delta

    def forward(
        self,
        x: Tensor,
        edge_types: Tensor,
        edge_index: Tensor,
        pred_steps: int = 1
    ) -> Tensor:
        """
        Make multistep predictions given the inputs and rel type predictions.
        :param x: node features [B, N, X_dim]
        :param edge_types: edge type predicted by the encoder # [B, E, edge_types]
        :param edge_index: edge index used by the encoder [2, E]. Defaults to fully meshed edge index.
        :param pred_steps: number of time steps to predict. Defaults to 1.
        :return: node features for next time step [B, N, pred_steps, X_dim]
        """
        x = x.transpose(1, 2).contiguous()  # [B, T, N, x_dim]
        B, T, N, x_dim = x.shape
        assert pred_steps <= T, "pred_steps exceeds available timesteps"

        predictions = []
        x_t = x[:, 0::pred_steps, :, :]
        for _ in range(pred_steps):
            x_t = self.single_step_forward(x_t, edge_index, edge_types)
            predictions.append(x_t)

        # Combine predictions
        output = torch.zeros(
            B, predictions[0].size(1) * pred_steps, N, x_dim, device=x.device
        )
        for i, p in enumerate(predictions):
            output[:, i::pred_steps, :, :] = p

        pred_all = output[:, :(T - 1), :, :]
        return pred_all.transpose(1, 2).contiguous()
