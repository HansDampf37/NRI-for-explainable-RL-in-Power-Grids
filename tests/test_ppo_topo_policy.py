import unittest
from typing import List

import grid2op
import hydra
from grid2op.Action import BaseAction
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from baselines.topo_policies_sb3 import TopoPolicyStablePPO
from baselines.train_stable_baseline import model_setup


class TestPPOTopoPolicy(unittest.TestCase):
    def setUp(self):
        with hydra.initialize(config_path="../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    "baseline=gnn_ppo",
                    "hydra.run.dir=."  # disable Hydra run dir creation
                ]
            )
            print(OmegaConf.to_yaml(cfg))
            model: PPO = model_setup(cfg)
            self.topo_policy = TopoPolicyStablePPO(model)

            self.env = grid2op.make(cfg.env.env_name)

    def test_ppo_topo_policy(self):
        k = 5
        actions = self.topo_policy.get_k_best_actions(self.env.current_obs, k)
        self.assertIsInstance(actions, List)
        self.assertIsInstance(actions[0], BaseAction)
        self.assertEqual(len(actions), k)