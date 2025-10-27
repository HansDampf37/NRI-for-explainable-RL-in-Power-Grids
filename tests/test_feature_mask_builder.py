import unittest

import grid2op
import numpy as np
from numpy.testing import assert_array_equal

from common import BusConnectivityGraphObsSpace
from nri.FeatureMaskBuilder import get_feature_mask


class TestFeatureMaskBuilder(unittest.TestCase):
    def setUp(self):
        env = grid2op.make("l2rpn_case14_sandbox")
        self.obs_space = BusConnectivityGraphObsSpace(env.observation_space)

    def test_empty_mask(self):
        mask = get_feature_mask(self.obs_space, [])
        expected = np.array([0] * self.obs_space.x_dim)
        assert_array_equal(mask, expected)

    def test_non_existent_feature(self):
        with self.assertRaises(KeyError):
            get_feature_mask(self.obs_space, ["non_existent"])


    def test_all_feature(self):
        mask = get_feature_mask(self.obs_space,self.obs_space.node_feature_names)
        expected = np.array([1] * self.obs_space.x_dim)
        assert_array_equal(mask, expected)
