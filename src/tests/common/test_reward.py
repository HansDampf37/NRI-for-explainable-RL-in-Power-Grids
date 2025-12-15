import unittest

import grid2op

from src.common.rewards import BaseWithBonus, MazeRLReward


class TestReward(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox", reward_class=MazeRLReward)
        self.env.reset()

    def test_reward(self):
        _, rew, _, _ = self.env.step(self.env.action_space.sample())
        self.assertTrue(0.0 <= rew <= 1)

class TestRewardWithBonus(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox", reward_class=BaseWithBonus)
        self.env.reset()

    def test_reward(self):
        _, rew, _, _ = self.env.step(self.env.action_space.sample())
        reward_class = BaseWithBonus()
        min_reward = reward_class.min_reward
        max_reward = reward_class.max_reward
        self.assertTrue(min_reward <= rew <= max_reward)