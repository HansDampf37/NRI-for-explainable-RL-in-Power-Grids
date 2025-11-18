import unittest
import torch
from torch import Tensor

from nri.Sampling import GumbelSoftmax


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.gumbel_softmax = GumbelSoftmax(tau=1.0)

    def test_output_shape(self):
        logits = torch.randn(3, 5)
        output = self.gumbel_softmax(logits)
        self.assertEqual(output.shape, logits.shape)

    def test_soft_sample_values_are_probabilities(self):
        logits = torch.randn(4, 6)
        output = self.gumbel_softmax(logits, hard=False)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

        row_sums = output.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4))

    def test_hard_sample_is_one_hot(self):
        logits = torch.randn(10, 7)
        output = self.gumbel_softmax(logits, hard=True)
        row_sums = output.sum(dim=-1)

        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
        self.assertTrue(torch.all((output == 0) | (output == 1)))

    def test_soft_sample_allows_gradients(self):
        logits = torch.randn(5, 8, requires_grad=True)
        output = self.gumbel_softmax(logits, hard=False)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)

    def test_hard_sample_allows_gradients(self):
        logits = torch.randn(4, 5, requires_grad=True)
        output = self.gumbel_softmax(logits, hard=True)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)

    def test_samples_are_stochastic(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5)
        out1 = self.gumbel_softmax(logits, hard=False)
        out2 = self.gumbel_softmax(logits, hard=False)
        self.assertFalse(torch.allclose(out1, out2))

    def test_sample_gumbel_distribution_mean(self):
        samples = self.gumbel_softmax.sample_gumbel(torch.Size([10000]))
        mean = samples.mean().item()
        # Mean of Gumbel(0,1) ≈ 0.5772
        self.assertAlmostEqual(mean, 0.577, delta=0.1)

    def test_gumbel_softmax_mean(self):
        num_draws = 1000
        p = 0.01
        dists = Tensor([p, 1-p]).repeat([num_draws, 1])
        logits = torch.log(dists)
        discrete = GumbelSoftmax(tau=0.1).forward(logits)
        expected_draws = num_draws * p
        self.assertTrue(abs(discrete[:, 0].sum().item() - expected_draws) < num_draws / 100)
