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
                x_dim=x_dim,
                hidden_dim=hidden_dim,
                e_dim=hidden_dim,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_edge_types)
        ])

        self.edge_node2node = EdgeNode2Node(
            x_dim=x_dim,
            e_dim=hidden_dim,
            hidden_dim=hidden_dim,
            x_out_dim=x_dim,
            dropout_prob=dropout_prob
        )

    def single_step_forward(
        self,
        x: Tensor,
        edge_types: Tensor,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform one step of message passing and node update. The input x is of shape [B, T, N, X_dim] where B is the
        batch size, T is an arbitrary number of timesteps, N is the number of nodes, and X_dim is the node feature dimension.
        Note that timesteps in T do not have to be successive. You can just consider B and T as batch dimensions. This
        function should predict next time steps for every timestep in the first two dimensions.

        :param x: node features [B, T, N, X_dim]
        :param edge_types: edge type predicted by the encoder # [B, E, edge_types]
        :param edge_index: edge index used by the encoder [2, E]. Defaults to fully meshed edge index.
        :return: node features for next time step [B, T, N, X_dim]
        """
        B, T, N, X_dim = x.shape
        _, E, NUM_EDGE_TYPES = edge_types.shape

        start_idx = 1 if self.skip_first else 0
        edge_index = edge_index if edge_index is not None else fully_connected_edge_index(N, x.device)
        edge_types = edge_types.view(B, 1, E, NUM_EDGE_TYPES)

        edge_properties_all = []
        for k in range(start_idx, self.num_edge_types):
            edge_properties = edge_types[..., k:k+1] * self.node2edge_list[k](x, edge_index)
            edge_properties_all.append(edge_properties)
        e = sum(edge_properties_all)

        delta = self.edge_node2node(x, e, edge_index)
        return x + delta

    def forward(
        self,
        x: Tensor,
        edge_types: Tensor,
        edge_index: Optional[Tensor] = None,
        pred_steps: int = 1
    ) -> Tensor:
        """
        Make multistep predictions given the inputs and edge type predictions.
        The inputs should be a tensor of shape [B, N, T, X_dim], where B is the batch size, N is the number of nodes,
        T is the number of timesteps, and X_dim is the dimension of the input features. This method will split the T
        dimension into chunks of size pred_steps. Then for each chunk, the single_step_forward method will predict the
        next time steps of the chunk.

        :param x: node features [B, T, N, X_dim]
        :param edge_types: edge type predicted by the encoder # [B, E, edge_types]
        :param edge_index: edge index used by the encoder [2, E]. Defaults to fully meshed edge index.
        :param pred_steps: number of time steps to predict. Defaults to 1.
        :return: node features for next time step [B, N, pred_steps, X_dim]
        """
        B, T, N, x_dim = x.shape
        edge_index = edge_index if edge_index is not None else fully_connected_edge_index(N, x.device)
        assert pred_steps <= T, "pred_steps exceeds available timesteps"
        x_t = x[:, 0::pred_steps, :, :]

        predictions = []
        for _ in range(pred_steps):
            x_t = self.single_step_forward(x_t, edge_types, edge_index)
            predictions.append(x_t)

        # Combine predictions
        T_plus_modulo_tail = predictions[0].size(1) * pred_steps
        output = torch.zeros(B, T_plus_modulo_tail, N, x_dim, device=x.device)
        for i, p in enumerate(predictions):
            output[:, i::pred_steps, :, :] = p

        return output[:, :(T-1), :, :]
