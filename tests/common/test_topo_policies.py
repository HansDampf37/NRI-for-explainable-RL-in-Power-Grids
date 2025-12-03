import unittest

import grid2op

from ra_agents.ppo import Sb3PPOTopologyPolicy, RAPPO


class TestPPOTopoPolicy(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.ppo = RAPPO.load("tests/resources/rappo.zip")
        self.topo_policy=Sb3PPOTopologyPolicy(self.ppo)

    def test_forward(self):
        actions = self.topo_policy.get_k_best_actions(self.env.current_obs, 10)
        self.assertEqual(len(actions), 10)