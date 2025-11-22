import tempfile
import unittest
from unittest import mock
import os

import hydra


class TestTrainingFunctions(unittest.TestCase):
    @staticmethod
    def _setup_cfg():
        with hydra.initialize(config_path="../../hydra_configs", version_base="1.3"):
            cfg = hydra.compose(config_name="config")
            test_train_cfg = hydra.compose(config_name="rl/train/test")
            test_eval_cfg = hydra.compose(config_name="rl/eval/test")

            cfg.rl.train = test_train_cfg.rl.train
            cfg.rl.eval = test_eval_cfg.rl.eval
            cfg.rl.ppo.sb3.n_steps = 2
            cfg.rl.dqn.sb3.learning_starts = 0
            return cfg

    @staticmethod
    def _assert_dir_not_empty(dirpath: str):
        assert os.path.exists(dirpath), f"Directory {dirpath} does not exist"
        assert os.path.isdir(dirpath), f"{dirpath} is not a directory"
        contents = os.listdir(dirpath)
        assert len(contents) > 0, f"Directory {dirpath} is empty"

    # --------------------------------------------------------------
    # 1. train_relations_aware_ppo
    # --------------------------------------------------------------
    def test_train_relations_aware_ppo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = tmpdir + "/eval"
            edge_probs_dir = tmpdir + "/edge_probs"
            models_dir = tmpdir + "/models"
            logs_dir = tmpdir + "/logs"

            with \
                mock.patch("ra_agents.ppo.train_RAPPO.EVAL_PATH", new=eval_dir), \
                mock.patch("ra_agents.ppo.train_RAPPO.EDGE_PROBS_PATH", new=edge_probs_dir), \
                mock.patch("ra_agents.ppo.train_RAPPO.MODELS_PATH", new=models_dir), \
                mock.patch("ra_agents.ppo.train_RAPPO.LOGS_PATH", new=logs_dir):

                from ra_agents.ppo import train_relations_aware_ppo
                cfg = self._setup_cfg()
                train_relations_aware_ppo(cfg)

            self._assert_dir_not_empty(eval_dir)
            self._assert_dir_not_empty(edge_probs_dir)
            self._assert_dir_not_empty(models_dir)
            self._assert_dir_not_empty(logs_dir)

    # --------------------------------------------------------------
    # 2. train_relations_aware_dqn
    # --------------------------------------------------------------
    def test_train_relations_aware_dqn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = tmpdir + "/eval"
            edge_probs_dir = tmpdir + "/edge_probs"
            models_dir = tmpdir + "/models"
            logs_dir = tmpdir + "/logs"

            with \
                mock.patch("ra_agents.dqn.train_RADQN.EVAL_PATH", new=eval_dir), \
                mock.patch("ra_agents.dqn.train_RADQN.EDGE_PROBS_PATH", new=edge_probs_dir), \
                mock.patch("ra_agents.dqn.train_RADQN.MODELS_PATH", new=models_dir), \
                mock.patch("ra_agents.dqn.train_RADQN.LOGS_PATH", new=logs_dir):

                from ra_agents.dqn import train_relations_aware_dqn
                cfg = self._setup_cfg()
                train_relations_aware_dqn(cfg)

            self._assert_dir_not_empty(eval_dir)
            self._assert_dir_not_empty(edge_probs_dir)
            self._assert_dir_not_empty(models_dir)
            self._assert_dir_not_empty(logs_dir)

    # --------------------------------------------------------------
    # 3. train_mlp_baseline (PPO)
    # --------------------------------------------------------------
    def test_train_mlp_baseline_ppo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = tmpdir + "/eval"
            models_dir = tmpdir + "/models"
            logs_dir = tmpdir + "/logs"

            with \
                mock.patch("ra_agents.ppo.train_MLP_baseline.EVAL_PATH", new=eval_dir), \
                mock.patch("ra_agents.ppo.train_MLP_baseline.MODELS_PATH", new=models_dir), \
                mock.patch("ra_agents.ppo.train_MLP_baseline.LOGS_PATH", new=logs_dir):

                from ra_agents.ppo import train_mlp_baseline
                cfg = self._setup_cfg()
                train_mlp_baseline(cfg)

            self._assert_dir_not_empty(eval_dir)
            self._assert_dir_not_empty(models_dir)
            self._assert_dir_not_empty(logs_dir)

    # --------------------------------------------------------------
    # 4. train_mlp_baseline (DQN)
    # --------------------------------------------------------------
    def test_train_mlp_baseline_dqn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = tmpdir + "/eval"
            models_dir = tmpdir + "/models"
            logs_dir = tmpdir + "/logs"

            with \
                mock.patch("ra_agents.dqn.train_MLP_baseline.EVAL_PATH", new=eval_dir), \
                mock.patch("ra_agents.dqn.train_MLP_baseline.MODELS_PATH", new=models_dir), \
                mock.patch("ra_agents.dqn.train_MLP_baseline.LOGS_PATH", new=logs_dir):

                from ra_agents.dqn import train_mlp_baseline as train_mlp_dqn_baseline
                cfg = self._setup_cfg()
                train_mlp_dqn_baseline(cfg)

            self._assert_dir_not_empty(eval_dir)
            self._assert_dir_not_empty(models_dir)
            self._assert_dir_not_empty(logs_dir)

    # --------------------------------------------------------------
    # 5. train_gnn_baseline (PPO)
    # --------------------------------------------------------------
    def test_train_gnn_baseline_ppo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = tmpdir + "/eval"
            models_dir = tmpdir + "/models"
            logs_dir = tmpdir + "/logs"

            with \
                mock.patch("ra_agents.ppo.train_GNN_baseline.EVAL_PATH", new=eval_dir), \
                mock.patch("ra_agents.ppo.train_GNN_baseline.MODELS_PATH", new=models_dir), \
                mock.patch("ra_agents.ppo.train_GNN_baseline.LOGS_PATH", new=logs_dir):

                from ra_agents.ppo import train_gnn_baseline
                cfg = self._setup_cfg()
                train_gnn_baseline(cfg)

            self._assert_dir_not_empty(eval_dir)
            self._assert_dir_not_empty(models_dir)
            self._assert_dir_not_empty(logs_dir)

    # --------------------------------------------------------------
    # 6. train_gnn_baseline (DQN)
    # --------------------------------------------------------------
    def test_train_gnn_baseline_dqn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = tmpdir + "/eval"
            models_dir = tmpdir + "/models"
            logs_dir = tmpdir + "/logs"

            with \
                mock.patch("ra_agents.dqn.train_GNN_baseline.EVAL_PATH", new=eval_dir), \
                mock.patch("ra_agents.dqn.train_GNN_baseline.MODELS_PATH", new=models_dir), \
                mock.patch("ra_agents.dqn.train_GNN_baseline.LOGS_PATH", new=logs_dir):

                from ra_agents.dqn import train_gnn_baseline as train_gnn_dqn_baseline
                cfg = self._setup_cfg()
                train_gnn_dqn_baseline(cfg)

            self._assert_dir_not_empty(eval_dir)
            self._assert_dir_not_empty(models_dir)
            self._assert_dir_not_empty(logs_dir)
