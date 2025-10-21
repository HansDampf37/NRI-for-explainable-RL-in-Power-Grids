from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as f

from nri.Decoder import Decoder
from nri.Encoder import Encoder
from nri.Sampling import GumbelSoftmax
from nri.utils import fully_connected_edge_index


class NRIModule(nn.Module):
    """
    The NRI module implements a special VAE that is trained to reconstruct next step feature vectors for markovian dynamical environments.
    As latent representation the encoder constructs a probability distribution for each edge over multiple edge types.
    The downstream decoder utilizes these distributions to reconstruct the next step feature vectors with a GNN.
    Each edge type implements its own MessagePassing instance. The respective message passing results are weighted by the probability of edge e being of the specific type. Therefore, edge types with low probability yield only insignificant contribution to the result.

    The probability distributions of the encoder are not directly sampled as we can not backpropagate through discrete sampling.
    Instead, we take the logits of the encoder and create the distribution over edge types as:

    softmax((logits + g)/tau) where g are samples from Gumbel (0,1)
    """

    def __init__(
            self,
            x_dim: int,
            hidden_dim: int,
            trajectory_length: int,
            num_edge_types: int = 2,
            pred_steps: int = 3,
            dropout_prob: float = 0.0,
            skip_first: bool = True
    ):
        """
        Constructs a NRI module.
        :param x_dim: number of input features for nodes
        :param hidden_dim: the dimensionality of hidden layers
        :param trajectory_length: the length of a single trajectory
        :param num_edge_types: the number of possible types for edges
        :param pred_steps: the number of prediction steps that the encoder should take
        :param dropout_prob: the dropout probability
        """
        super(NRIModule, self).__init__()
        self.encoder = Encoder(
            x_dim=x_dim,
            trajectory_length=trajectory_length,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            dropout_prob=dropout_prob
        )
        self.decoder = Decoder(
            x_dim=x_dim,
            num_edge_types=num_edge_types,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            skip_first=skip_first
        )
        self.gumbel_softmax = GumbelSoftmax()
        self.pred_steps = pred_steps
        self.num_edge_types = num_edge_types
        self.skip_first = skip_first


    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the NRI module.
        Pushes the input-tensor x of shape [(B), T, N, X_dim] through the encoder to get p(z|x).
        Applies Gumbel-Softmax to create approximately one-hot distributions p_one_hot.
        Pushes x with p_one_hot through the decoder to predict next time steps.

        :param x: Input tensor of shape [(B), T, N, X_dim]
        :param edge_index: node adjacency [2, E]. Only latent edges that are included in this argument are detected. Per default this is fully meshed.
        :return: Predicted next step feature vector of shape [(B), T, N, X_dim] and p(z|x) of shape [(B), E, num_edge_types]
        """
        T, N, X_dim = x.shape[-3:]
        edge_index = edge_index if edge_index is not None else fully_connected_edge_index(N, x.device)
        encoder_logits = self.encoder.forward(x, edge_index)

        p_z_given_x = f.softmax(encoder_logits, dim=-1)
        p_one_hot = self.gumbel_softmax(encoder_logits)

        predictions = self.decoder.forward(x, p_one_hot, edge_index, self.pred_steps)
        return predictions, p_z_given_x

    def get_latent_edges(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """
        Forward passes x through the encoder to get p(z|x) and sample from it using the hard Gumbel-Softmax.

        :param x: Input tensor of shape [(B), T, N, X_dim]
        :param edge_index: Node adjacency [2, E]
        :return: Batched edge index of shape [(B), 3, E] where dimension 1 includes receiver, sender, and type
        """
        added_batch_dim = False
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1, T, N, X_dim]
            added_batch_dim = True

        B, T, N, X_dim = x.shape
        edge_index = edge_index if edge_index is not None else fully_connected_edge_index(N, x.device)
        _, E = edge_index.shape

        encoder_logits: Tensor = self.encoder(x, edge_index)  # [B, E, K]
        p_one_hot: Tensor = self.gumbel_softmax(encoder_logits, hard=True)  # [B, E, K]

        batched_edge_index = edge_index.unsqueeze(0).expand(B, -1, -1)  # [B, 2, E]
        edge_types = torch.argmax(p_one_hot, dim=-1)  # [B, E]
        full_edge_index = torch.concatenate([batched_edge_index, edge_types.view(B, 1, E)], dim=1)

        if added_batch_dim:
            return full_edge_index.squeeze(0)  # shape: [3, E]
        else:
            return full_edge_index  # shape: [B, 3, E]
