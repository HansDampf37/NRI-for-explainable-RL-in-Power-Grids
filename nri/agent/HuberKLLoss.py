import math
from typing import Optional, Tuple, Union

import numpy.typing as npt
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class HuberKLLoss(nn.Module):
    """
    HuberKLL is defined as:
    L(x, target) = (α * E_{z ~ q_φ(.|x), y ~ p_θ(y|x,z)}[huber(y, target)] + β * KL[q_φ(z|x)||p_θ(z)]) / (α + β)

    The encoder q_φ(z|x) returns a factorized distribution of z_ij.
    The NRI_informed_GNN p_θ(y|x,z) makes a prediction based on input x and z_ij
    The prior p_θ(z) is a distribution that the posterior distribution q_φ(z|x) is pushed towards.
    """

    def __init__(self, prior: Optional[npt.NDArray] = None, alpha: float = 1.0, beta: float = 1.0):
        """
        Constructor

        :param prior: The prior assumption about the distribution
        :param alpha: weight for the negative Huber loss
        :param beta: weight for the KL-Divergence
        """
        super().__init__()
        self.eps = 1e-10  # for numerical stability
        self.alpha = alpha
        self.beta = beta

        if prior is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            prior = torch.from_numpy(prior).to(dtype=torch.float32, device=device)
            prior = prior / prior.sum(dim=-1, keepdim=True)
            self.register_buffer('prior', prior)
        else:
            self.prior = None

    def forward(self, predictions: Tensor, target: Tensor, posterior_probs: Tensor, with_huber_kl: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute the HuberKLL loss for the inputs. Predictions should be close to target and posterior_probs should be close to the prior.

        :param predictions: predictions by the model
        :param target: the targets
        :param posterior_probs: the distributions predicted by the encoder
        :param with_huber_kl: if this is true the huberLoss and kl are also returned
        :return: the HuberKLLoss (and huber, kl if with_huber_kl == True)
        """
        huber = F.smooth_l1_loss(predictions, target)
        kl = self.kl_divergence_to_prior(posterior_probs)
        loss = (self.alpha * huber + self.beta * kl) / (self.alpha + self.beta)
        return (loss, huber, kl) if with_huber_kl else loss


    def kl_divergence_to_prior(self, posterior_probs: Tensor) -> Tensor:
        """
        Computes the kl divergence of the specified tensor to a uniform distribution.
        :param posterior_probs: posterior distribution(s)
        """
        if self.prior is None:
            # Negative entropy: -H(q) = sum q * log q
            neg_entropy = (posterior_probs * torch.log(posterior_probs)).sum(dim=-1)
            constant_term = math.log(posterior_probs.shape[-1])
            return neg_entropy.mean() + constant_term
        else:
            # KL(q || p): sum q * log(q / p) = sum q * (log(q) - log(p))
            prior = self.prior.unsqueeze(0)
            kl = (posterior_probs * (torch.log(posterior_probs + self.eps) - torch.log(prior + self.eps))).sum(dim=-1)
            return kl.mean()
