import torch
from torch import nn, Tensor

from common.MLP import MLP


class Decoder(nn.Module):
    """
    MLP-based interaction network decoder (Kipf et al., 2018),
    refactored to use edge indices and modular MLP blocks.
    """

    def __init__(
        self,
        node_dim: int,
        edge_types: int,
        msg_hid: int,
        msg_out: int,
        hidden_dim: int,
        dropout_prob: float = 0.0,
        skip_first: bool = False,
    ):
        """
        Args:
            node_dim: Dimensionality of node features (e.g., pos+vel)
            edge_types: Number of edge types (latent relations)
            msg_hid: Hidden dimension of message MLPs
            msg_out: Output dimension of message MLPs
            hidden_dim: Hidden dimension of node update MLP
            dropout_prob: Dropout probability
            skip_first: If True, skip first edge type (often 'no-edge')
        """
        super().__init__()

        self.edge_types = edge_types
        self.skip_first = skip_first
        self.dropout_prob = dropout_prob
        self.msg_out = msg_out

        # One MLP per edge type for message computation
        self.message_mlps = nn.ModuleList([
            MLP(
                input_features=2 * node_dim,
                hidden_dim=msg_hid,
                output_features=msg_out,
                dropout_prob=dropout_prob
            )
            for _ in range(edge_types)
        ])

        # Node update MLP (aggregates messages + node features)
        self.update_mlp = MLP(
            input_features=node_dim + msg_out,
            hidden_dim=hidden_dim,
            output_features=node_dim,
            dropout_prob=dropout_prob,
        )

    def single_step_forward(
        self,
        x: Tensor,              # [B, N, F]
        edge_index: Tensor,     # [2, E]
        rel_type: Tensor        # [B, E, edge_types]
    ) -> Tensor:
        """
        One step of message passing and node update.

        Args:
            x: Node features [B, N, F]
            edge_index: Edge connectivity [2, E]
            rel_type: Edge-type probabilities [B, E, edge_types]

        Returns:
            Updated node features [B, N, F]
        """
        B, N, F = x.shape
        senders, receivers = edge_index
        E = edge_index.size(1)

        # Gather node features for each edge
        x_send = x[:, senders, :]   # [B, E, F]
        x_recv = x[:, receivers, :] # [B, E, F]
        edge_input = torch.cat([x_send, x_recv], dim=-1)  # [B, E, 2F]

        # Compute weighted messages per edge type
        msg_all = torch.zeros(B, E, self.msg_out, device=x.device)
        start_idx = 1 if self.skip_first else 0

        for k in range(start_idx, self.edge_types):
            msg = self.message_mlps[k](edge_input)        # [B, E, msg_out]
            msg = msg * rel_type[..., k:k+1]              # weight by edge-type prob
            msg_all += msg

        # Aggregate messages to nodes (sum incoming messages)
        agg_msg = torch.zeros(B, N, self.msg_out, device=x.device)
        agg_msg.index_add_(1, receivers, msg_all)

        # Concatenate node features + aggregated message
        node_input = torch.cat([x, agg_msg], dim=-1)

        # Node update (predicts delta)
        delta = self.update_mlp(node_input)

        # Residual connection: next_state = current + delta
        return x + delta

    def forward(
        self,
        inputs: Tensor,       # [B, N, T, F]
        rel_type: Tensor,     # [B, E, edge_types]
        edge_index: Tensor,   # [2, E]
        pred_steps: int = 1
    ) -> Tensor:
        """
        Roll out multi-step predictions.

        Args:
            inputs: Node features over time [B, N, T, F]
            rel_type: Edge-type probabilities [B, E, edge_types]
            edge_index: Edge connectivity [2, E]
            pred_steps: Number of time steps to predict

        Returns:
            Predicted node features [B, N, T-1, F]
        """
        inputs = inputs.transpose(1, 2).contiguous()  # [B, T, N, F]
        B, T, N, F = inputs.shape
        assert pred_steps <= T, "pred_steps exceeds available timesteps"

        preds = []
        x_t = inputs[:, 0::pred_steps, :, :]

        for _ in range(pred_steps):
            x_t = self.single_step_forward(x_t, edge_index, rel_type)
            preds.append(x_t)

        # Combine predictions
        output = torch.zeros(
            B, preds[0].size(1) * pred_steps, N, F, device=inputs.device
        )
        for i, p in enumerate(preds):
            output[:, i::pred_steps, :, :] = p

        pred_all = output[:, :(T - 1), :, :]
        return pred_all.transpose(1, 2).contiguous()
