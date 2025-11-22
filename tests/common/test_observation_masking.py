import unittest

import grid2op
import numpy as np
from numpy.testing import assert_array_equal

from common.graph_structured_observation_space import BusConnectivityGraphObsSpace, NODES
from common.mask_observations import get_feature_mask


class TestObservationMasking(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.obs_space = BusConnectivityGraphObsSpace(self.env.observation_space)

    def test_empty_mask(self):
        mask = get_feature_mask(self.obs_space, [])
        expected = np.array([0] * self.obs_space.x_dim).astype(bool)
        assert_array_equal(mask, expected)

    def test_non_existent_feature(self):
        with self.assertRaises(KeyError):
            get_feature_mask(self.obs_space, ["non_existent"])

    def test_all_feature(self):
        mask = get_feature_mask(self.obs_space, self.obs_space.node_feature_names)
        expected = np.array([1] * self.obs_space.x_dim).astype(bool)
        assert_array_equal(mask, expected)

    def test_masking(self):
        mask = get_feature_mask(self.obs_space, ["rho"])
        obs = self.obs_space.to_gym(self.env.reset())[NODES]
        masked_obs = obs[..., mask]
        self.assertEqual(masked_obs.shape[-1], 1)
