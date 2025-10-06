import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable


class GumbelSoftmax(nn.Module):
    def __init__(self, tau: float=1.0, eps: float=1e-10):
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
        Sample Tensor according to gumbel distribution
        """
        uniform_samples = torch.rand(shape).float()
        return -torch.log(self.eps - torch.log(uniform_samples + self.eps))

    def forward(self, x: Tensor) -> Tensor:
        """
        Draw a sample from the Gumbel-Softmax distribution.
        :param x: [..., n_class] unnormalized log-probs
        """
        gumbel_noise = self.sample_gumbel(x.size()).to(device=x.device)
        return F.softmax(x + Variable(gumbel_noise) / self.tau, dim=-1)
