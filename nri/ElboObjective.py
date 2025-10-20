import math
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor


class ElboLoss(nn.Module):
    """
    ELBO objective is defined as:
    L = α * E_{q_φ(z|x)}[log p_θ(x|z)] − β * KL[q_φ(z|x)||p_θ(z)]

    The encoder q_φ(z|x) returns a factorized distribution of z_ij.
    The decoder reconstructs the input x with p_θ(x|z).
    The prior p_θ(z) is a distribution that the posterior distribution q_φ(z|x) is pushed towards.

    This objective should be maximized. This module therefore implements the ELBO loss defined as -L
    """

    def __init__(self, prior: Optional[np.ndarray] = None, variance: float = 1.0, alpha: float = 1.0, beta: float = 1.0):
        """
        Constructor

        @param prior: The prior assumption about the distribution
        @param variance: Fixed variance of the output distribution that are used to sample the decoders output
        @param alpha: weight for the negative log-likelihood loss
        @param beta: weight for the KL-Divergence
        """
        super().__init__()
        self.eps = 1e-10  # for numerical stability
        self.variance = variance
        self.alpha = alpha
        self.beta = beta

        if prior is not None:
            prior = torch.tensor(prior, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
            prior = prior / prior.sum()
            self.register_buffer('prior', prior)
        else:
            self.prior = None

    def forward(self, predictions: Tensor, target: Tensor, posterior_probs: Tensor) -> Tensor:
        """
        Compute the ELBO loss (- ELBO objective) for the inputs. Predictions should be close to target and posterior_probs
        should be close to the prior.

        @param predictions: the VAEs output
        @param target: the VAEs target (input)
        @param posterior_probs: the distributions predicted by the encoder
        @return: the ELBO loss
        """
        nll = self.neg_log_likelihood(predictions, target)
        kl = self.kl_divergence_to_prior(posterior_probs)
        return nll + kl

    def neg_log_likelihood(self, predictions: Tensor, target: Tensor) -> Tensor:
        """
        Computes the negative log-likelihood under Gaussian assumption. For gaussian distributions this is equivalent
        to MSE scaled by variance:
        log p(x|z) ∝ (x - x_hat)^2 / (2 * variance)

        :param predictions: predictions tensor
        :param target: target tensor
        :return: Mean NLL over batch.
        """
        mse_term = ((predictions - target) ** 2) / (2 * self.variance)
        D = predictions.shape[-1]
        constant_term = 0.5 * D * math.log(2 * math.pi * self.variance)
        return self.alpha * (mse_term.sum(dim=-1).mean() + constant_term)

    def kl_divergence_to_prior(self, posterior_probs: Tensor) -> Tensor:
        """
        Computes the kl divergence of the specified tensor to a uniform distribution.
        :param posterior_probs: posterior distribution(s)
        """
        if self.prior is None:
            # Negative entropy: -H(q) = sum q * log q
            neg_entropy = (posterior_probs * torch.log(posterior_probs)).sum(dim=-1)
            constant_term = math.log(posterior_probs.shape[-1])
            return self.beta * (neg_entropy.mean() + constant_term)
        else:
            # KL(q || p): sum q * log(q / p) = sum q * (log(q) - log(p))
            prior = self.prior.unsqueeze(0)
            kl = (posterior_probs * (torch.log(posterior_probs + self.eps) - torch.log(prior + self.eps))).sum(dim=-1)
            return self.beta * (kl.mean())
