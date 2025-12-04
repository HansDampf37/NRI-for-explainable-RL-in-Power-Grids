import unittest

import grid2op

from ra_agents.dqn import RADQN, Sb3DQNTopologyPolicy
from ra_agents.ppo import Sb3PPOTopologyPolicy, RAPPO


class TestPPOTopoPolicy(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")

    def test_forward_ppo(self):
        ppo = RAPPO.load("tests/resources/rappo.zip")
        topo_policy = Sb3PPOTopologyPolicy(ppo)
        actions = topo_policy.get_k_best_actions(self.env.current_obs, 10)
        self.assertEqual(len(actions), 10)

    def test_forward_dqn(self):
        dqn = RADQN.load("tests/resources/radqn.zip")
        topo_policy_dqn = Sb3DQNTopologyPolicy(dqn)
        actions = topo_policy_dqn.get_k_best_actions(self.env.current_obs, 10)
        self.assertEqual(len(actions), 10)