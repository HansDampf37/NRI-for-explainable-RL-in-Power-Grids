# ...existing code...
import unittest
import torch
from nri.agent.dqn.HuberKLLoss import HuberKLLoss


class TestHuberKLLoss(unittest.TestCase):
    def test_huber_kl_loss_without_prior(self):
        # simple check: uniform posterior, prior none
        loss_module = HuberKLLoss(prior=None, alpha=2.0, beta=3.0)
        predictions = torch.zeros((2, 3))
        target = torch.ones_like(predictions)
        posterior = torch.softmax(torch.randn(2, 4), dim=-1)  # shape [2, K]

        loss, huber, kl = loss_module.forward(predictions, target, posterior, with_huber_kl=True)
        # loss should equal weighted average
        expected_loss = (2.0 * huber + 3.0 * huber.abs() * kl) / (2.0 + 3.0)
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=6)

    def test_huber_kl_loss_with_prior(self):
        prior = torch.tensor([0.6, 0.4])
        loss_module = HuberKLLoss(prior=prior, alpha=1.0, beta=1.0)
        predictions = torch.tensor([[0.0], [1.0]])
        target = torch.tensor([[0.5], [0.9]])
        posterior = torch.tensor([[0.9, 0.1], [0.5, 0.5]])

        loss, huber, kl = loss_module.forward(predictions, target, posterior, with_huber_kl=True)
        # compute manual KL: mean over batch of sum q*(log q - log p)
        eps = 1e-10
        manual_kl = (posterior * (torch.log(posterior + eps) - torch.log(loss_module.prior + eps))).sum(dim=-1).mean()
        self.assertAlmostEqual(kl.item(), manual_kl.item(), places=6)

    def test_huber_kl_loss_zero_posterior_entries_should_be_handled(self):
        loss_module = HuberKLLoss(prior=None)
        posterior = torch.tensor([[1.0, 0.0, 0.0]])
        kl = loss_module.kl_divergence_to_prior(posterior)
        self.assertTrue(torch.isfinite(kl))


if __name__ == '__main__':
    unittest.main()

