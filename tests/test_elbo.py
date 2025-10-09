import math
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
        loss = elbo.forward(x, x + 1, posterior)
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
        loss = ElboLoss().forward(predictions, x[:, 1:, :, :], posterior)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0)

    def test_zero_loss_when_prediction_matches_target(self):
        x = torch.randn(8, 10, 5)
        posterior = torch.full((8, 5), 1.0 / 5)  # uniform
        elbo = ElboLoss(variance=1/(2*math.pi))
        loss = elbo.forward(x, x, posterior)
        # NLL should be 0 (constant loss is 0 due to choice of variance), KL should be 0
        self.assertEqual(loss.item(), 0)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_kl_with_known_prior(self):
        posterior = torch.tensor([
            [0.5, 0.5],
            [0.9, 0.1],
        ])
        prior = np.array([0.5, 0.5])
        elbo = ElboLoss(prior=prior)

        kl = elbo.kl_divergence_to_prior(posterior)
        expected_kl_1 = 0  # for [0.5, 0.5]
        expected_kl_2 = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)
        expected_kl = (expected_kl_1 + expected_kl_2) / 2
        self.assertAlmostEqual(kl.item(), expected_kl, places=4)

    def test_numerical_stability_with_low_probs(self):
        posterior = torch.tensor([
            [1e-10, 1 - 1e-10],
            [1 - 1e-10, 1e-10]
        ])
        elbo = ElboLoss()
        kl = elbo.kl_divergence_to_prior(posterior)
        self.assertFalse(torch.isnan(kl))
        self.assertGreaterEqual(kl.item(), 0)

    def test_loss_increases_with_worse_prediction(self):
        x = torch.zeros(10, 5)
        posterior = torch.full((10, 5), 1.0 / 5)
        elbo = ElboLoss()

        good_pred = x.clone()
        bad_pred = x + 5.0  # large error

        loss_good = elbo(good_pred, x, posterior)
        loss_bad = elbo(bad_pred, x, posterior)
        self.assertLess(loss_good.item(), loss_bad.item())

    def test_shape_mismatch_raises(self):
        elbo = ElboLoss()
        x = torch.randn(10, 5)
        y = torch.randn(12, 5)  # mismatch
        posterior = torch.full((10, 5), 1.0 / 5)
        with self.assertRaises(RuntimeError):
            _ = elbo(x, y, posterior)

    def test_custom_variance(self):
        x = torch.zeros(4, 5)
        pred = torch.ones_like(x)
        posterior = torch.full((4, 5), 1.0 / 5)

        elbo_high_var = ElboLoss(variance=10.0)
        elbo_low_var = ElboLoss(variance=0.1)

        loss_high_var = elbo_high_var(pred, x, posterior)
        loss_low_var = elbo_low_var(pred, x, posterior)

        # Higher variance means lower NLL term
        self.assertLess(loss_high_var.item(), loss_low_var.item())
