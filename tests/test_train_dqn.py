import unittest

import grid2op
from grid2op.Action import BaseAction, TopologySetAction

from baselines.train_stable_dqn_baseline import model_setup, build_agent


class TestTrainDQN(unittest.TestCase):
    def test_model_setup(self):
        dqn = model_setup()
        dqn.learn(10)

    def test_build_agent(self):
        agent = build_agent(env_name="l2rpn_case14_sandbox", load_weights_from=None)
        agent.topology_policy.get_k_best_actions = lambda _1, _2: []
        env = grid2op.make("l2rpn_case14_sandbox")
        agent_action = agent.act(env.current_obs, 0)
        self.assertIsInstance(agent_action, BaseAction)
