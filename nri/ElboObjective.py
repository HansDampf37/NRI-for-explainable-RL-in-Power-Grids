from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor

from nri.utils import uniform_dist


class ElboLoss(nn.Module):
    """
    ELBO objective is defined as:
    L = E_{q_φ(z|x)}[log p_θ(x|z)] − KL[q_φ(z|x)||p_θ(z)]
    The encoder q_φ(z|x) returns a factorized distribution of z_ij.
    The decoder reconstructs the input x with p_θ(x|z).
    The prior p_θ(z) is a distribution that the posterior distribution q_φ(z|x) is pushed towards.

    This objective should be maximized. This module therefore implements the ELBO loss defined as -L
    """
    def __init__(self, prior: Optional[np.ndarray] = None, variance: float = 1.0, eps: float = 1e-10):
        super(ElboLoss, self).__init__()
        self.variance = variance
        self.eps = eps
        self.prior = prior

    def forward(self, predictions: Tensor, target: Tensor, posterior_probs: Tensor) -> Tensor:
        reconstruction = self.reconstruction_loss(predictions, target)
        kl_to_prior = self.kl_loss(posterior_probs)

        return reconstruction + kl_to_prior

    def reconstruction_loss(self, predictions: Tensor, target: Tensor) -> Tensor:
        """
        Computes the reconstruction loss for a batch of predictions and target. The loss is normalized by the amount of
        samples in the batch.
        :param predictions: predictions tensor
        :param target: target tensor
        :return: reconstruction loss
        """
        reconstruction_loss = torch.sum((predictions - target) ** 2) / (2 * self.variance)
        if len(predictions.shape) > 1:
            # normalize by batch size
            vectors_in_batch = np.prod(predictions.shape[0:-1])
            reconstruction_loss /= vectors_in_batch
        return reconstruction_loss

    def kl_loss(self, posterior_probs: Tensor) -> Tensor:
        """
        Computes the kl divergence of the specified tensor to a uniform distribution. According to wikipedia this
        is equivalent to the entropy of the distribution.
        :param posterior_probs: posterior distribution(s)
        """
        if self.prior is None:
            prior = uniform_dist(posterior_probs.size(-1)).to(device=posterior_probs.device)
        else:
            prior = self.prior

        kl_div = (posterior_probs * torch.log((posterior_probs + self.eps) / prior)).sum(dim=-1)
        return kl_div.mean()
