import os
import tempfile
import unittest
from pathlib import Path

import hydra
import numpy as np
from grid2op.Agent import DoNothingAgent

from src.common.constants import enable_test_mode
from src.common.graph_structured_observation_space import NODES
from src.nri.create_dataset import generate_dataset
from src.nri.train_nri import main as train_nri
from src.nri.utils import get_env, prior_from_env
from src.ra_agents.dqn import train_gnn_baseline as train_gnn_dqn_baseline, RADQN, HuberKLLoss
from src.ra_agents.dqn import train_mlp_baseline as train_mlp_dqn_baseline
from src.ra_agents.dqn import train_relations_aware_dqn
from src.ra_agents.ppo import train_gnn_baseline, RAPPO
from src.ra_agents.ppo import train_mlp_baseline
from src.ra_agents.ppo import train_relations_aware_ppo


class TestTrainingFunctions(unittest.TestCase):
    def setUp(self):
        enable_test_mode()

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
            cfg.rl.dqn.sb3.train_freq = 1
            cfg.rl.dqn.sb3.batch_size = 1
            cfg.nri.train.num_epochs = 1
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
        from src.common.constants import EVAL_PATH, EDGE_PROBS_PATH, MODELS_PATH, LOGS_PATH
        cfg = self._setup_cfg()
        train_relations_aware_ppo(cfg)

        self._assert_dir_not_empty(EVAL_PATH)
        self._assert_dir_not_empty(EDGE_PROBS_PATH)
        self._assert_dir_not_empty(MODELS_PATH)
        self._assert_dir_not_empty(LOGS_PATH)

    # --------------------------------------------------------------
    # 2. train_relations_aware_dqn
    # --------------------------------------------------------------
    def test_train_relations_aware_dqn(self):
        from src.common.constants import EVAL_PATH, EDGE_PROBS_PATH, MODELS_PATH, LOGS_PATH
        cfg = self._setup_cfg()
        train_relations_aware_dqn(cfg)

        self._assert_dir_not_empty(EVAL_PATH)
        self._assert_dir_not_empty(EDGE_PROBS_PATH)
        self._assert_dir_not_empty(MODELS_PATH)
        self._assert_dir_not_empty(LOGS_PATH)

    # --------------------------------------------------------------
    # 3. train_mlp_baseline (PPO)
    # --------------------------------------------------------------
    def test_train_mlp_baseline_ppo(self):
        from src.common.constants import EVAL_PATH, MODELS_PATH, LOGS_PATH
        cfg = self._setup_cfg()
        train_mlp_baseline(cfg)

        self._assert_dir_not_empty(EVAL_PATH)
        self._assert_dir_not_empty(MODELS_PATH)
        self._assert_dir_not_empty(LOGS_PATH)

    # --------------------------------------------------------------
    # 4. train_mlp_baseline (DQN)
    # --------------------------------------------------------------
    def test_train_mlp_baseline_dqn(self):
        from src.common.constants import EVAL_PATH, MODELS_PATH, LOGS_PATH
        cfg = self._setup_cfg()
        train_mlp_dqn_baseline(cfg)

        self._assert_dir_not_empty(EVAL_PATH)
        self._assert_dir_not_empty(MODELS_PATH)
        self._assert_dir_not_empty(LOGS_PATH)

    # --------------------------------------------------------------
    # 5. train_gnn_baseline (PPO)
    # --------------------------------------------------------------
    def test_train_gnn_baseline_ppo(self):
        from src.common.constants import EVAL_PATH, MODELS_PATH, LOGS_PATH
        cfg = self._setup_cfg()
        train_gnn_baseline(cfg)

        self._assert_dir_not_empty(EVAL_PATH)
        self._assert_dir_not_empty(MODELS_PATH)
        self._assert_dir_not_empty(LOGS_PATH)

    # --------------------------------------------------------------
    # 6. train_gnn_baseline (DQN)
    # --------------------------------------------------------------
    def test_train_gnn_baseline_dqn(self):
        from src.common.constants import EVAL_PATH, MODELS_PATH, LOGS_PATH
        cfg = self._setup_cfg()
        train_gnn_dqn_baseline(cfg)

        self._assert_dir_not_empty(EVAL_PATH)
        self._assert_dir_not_empty(MODELS_PATH)
        self._assert_dir_not_empty(LOGS_PATH)

    # --------------------------------------------------------------
    # 7. train_nri
    # --------------------------------------------------------------
    def test_train_nri(self):
        from src.common.constants import EDGE_PROBS_PATH, MODELS_PATH, LOGS_PATH
        cfg = self._setup_cfg()
        env = get_env(cfg)
        ds = generate_dataset(cfg.nri.train.batch_size, 10, DoNothingAgent(env._g2op_env.action_space), env._g2op_env, env.observation_space)
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = Path(tmpdir, f'node_features.npy')
            np.save(ds_path, ds[NODES])
            cfg.nri.train.training_dataset_path = ds_path
            cfg.nri.train.testing_dataset_path = ds_path
            cfg.nri.model.trajectory_length = 10
            train_nri(cfg)

        self._assert_dir_not_empty(EDGE_PROBS_PATH)
        self._assert_dir_not_empty(MODELS_PATH)
        self._assert_dir_not_empty(LOGS_PATH)

    def test_load_relations_aware_ppo(self):
        cfg = self._setup_cfg()
        # assure loading the algorithm works
        rappo = RAPPO.load("src/tests/resources/rappo.zip")
        rappo.set_prior(prior_from_env(1.0, get_env(cfg)))

    def test_load_relations_aware_dqn(self):
        cfg = self._setup_cfg()
        # assure loading the algorithm works
        radqn = RADQN.load("src/tests/resources/radqn.zip")
        radqn.set_loss_function(HuberKLLoss(prior_from_env(1.0, get_env(cfg))))


