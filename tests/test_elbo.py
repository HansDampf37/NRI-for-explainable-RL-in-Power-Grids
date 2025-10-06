import unittest

import numpy as np
import torch
from torch import Tensor

from nri.ElboObjective import ElboLoss
from nri.NRI import NRIModule


class TestElbo(unittest.TestCase):
    def test_elbo(self):
        x = torch.randn(10, 20, 30)
        posterior = torch.nn.functional.softmax(x, dim=2)
        prior = torch.nn.functional.softmax(Tensor(np.array([1] * 30)), dim=0)
        elbo = ElboLoss(prior)
        loss = elbo.forward(x, x+1, posterior)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_nri_module_compatibility(self):
        trajectory_length = 100
        x_dim = 10
        nri_module = NRIModule(
            x_dim,
            10,
            trajectory_length,
            2,
            3,
            0.0
        )
        x = torch.randn((100, trajectory_length, 6, x_dim))
        predictions, posterior = nri_module.forward(x)
        loss = ElboLoss().forward(predictions, x, posterior)
        self.assertEqual(loss.shape, torch.Size([]))