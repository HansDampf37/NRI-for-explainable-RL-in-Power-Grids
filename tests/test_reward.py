import unittest

import grid2op

from common.rewards import MazeRLReward


class TestReward(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox", reward_class=MazeRLReward)
        self.env.reset()

    def test_reward(self):
        _, rew, _, _ = self.env.step(self.env.action_space.sample())
        self.assertTrue(0.0 <= rew <= 1)
