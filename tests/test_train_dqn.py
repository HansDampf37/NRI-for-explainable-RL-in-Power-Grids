import unittest

import grid2op
import hydra
from grid2op.Action import BaseAction
from omegaconf import DictConfig

from baselines.train_stable_dqn_baseline import model_setup, build_agent


class TestTrainDQN(unittest.TestCase):
    def test_model_setup(self):
        with hydra.initialize(config_path="../hydra_configs", version_base="1.3"):
            env_cfg = hydra.compose(config_name="config")
            dqn = model_setup(env_cfg)
            dqn.learn(10)

    def test_build_agent(self):
        with hydra.initialize(config_path="../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(config_name="config")
            agent = build_agent(cfg, load_weights_from=None)
            agent.topology_policy.get_k_best_actions = lambda _1, _2: []
            env = grid2op.make("l2rpn_case14_sandbox")
            agent_action = agent.act(env.current_obs, 0)
            self.assertIsInstance(agent_action, BaseAction)
