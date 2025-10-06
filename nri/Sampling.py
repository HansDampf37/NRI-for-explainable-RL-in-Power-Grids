import torch
import torch.nn.functional as f
from torch import nn, Tensor


class GumbelSoftmax(nn.Module):
    """
    Differentiable sampling from a categorical distribution using the Gumbel-Softmax trick.
    """
    def __init__(self, tau: float=1.0, eps: float = 1e-10):
        """
        Creates a gumbel-softmax module
        :param tau: non-negative scalar temperature
        :param eps: parameter for gumbel sampling, prevents NaN results do tu log(0)
        """
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.eps = eps

    def sample_gumbel(self, shape: torch.Size) -> Tensor:
        """
        Sample Gumbel noise from Gumbel(0, 1)
        """
        uniform_samples = torch.rand(shape).float()
        return -torch.log(self.eps - torch.log(uniform_samples + self.eps))

    def forward(self, x: Tensor, hard: bool = False) -> Tensor:
        """
        Draw a differentiable sample from the Gumbel-Softmax distribution.
        :param x: logits [..., n_classes]
        :param hard: if True, return one-hot (hard) samples with straight-through gradients
        """
        gumbel_noise = self.sample_gumbel(x.size()).to(device=x.device)
        y = f.softmax((x + gumbel_noise) / self.tau, dim=-1)

        if hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        return y
