import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import grid2op
import hydra
from grid2op.Agent import DoNothingAgent
from stable_baselines3 import DQN

from src.common.baseline_agent import evaluate_agent, evaluate_sb3_alg
from src.common.env import G2OpGymEnv
from src.ra_agents.utils import evaluate, get_env_mlp_baseline


class TestAgentEvaluation(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.agent = DoNothingAgent(self.env.action_space)

    def test_evaluate_agent(self):
        with TemporaryDirectory() as tmpdir:
            evaluate_agent(
                env=self.env,
                agent=self.agent,
                path_results=Path(tmpdir),
                num_episodes=1,
                max_episode_length=100
            )
            # assert that tmpdir folder is not empty
            self.assertTrue(len(os.listdir(tmpdir)) != 0)


    def test_evaluate_sb3_alg(self):
        gym_env = G2OpGymEnv()
        dqn = DQN(env=gym_env, policy="MlpPolicy")

        with TemporaryDirectory() as tmpdir:
            path_results = Path(tmpdir)
            evaluate_sb3_alg(
                alg=dqn,
                env=gym_env,
                path_results=path_results,
                num_episodes=1,
            )
            # assert that tmpdir folder is not empty
            self.assertTrue(len(os.listdir(tmpdir)) != 0)

    def test_evaluate(self):
        with hydra.initialize(config_path="../../../configs", version_base="1.3"):
            cfg = hydra.compose(config_name="configs")
            cfg.rl.eval.final.nb_episodes = 1
            cfg.rl.eval.final.max_episode_length = 10
            with TemporaryDirectory() as tmpdir:
                env = get_env_mlp_baseline(cfg, cfg.env.name)
                dqn = DQN(env=env, policy="MlpPolicy")
                evaluate(
                    algorithm=dqn,
                    env_creation=get_env_mlp_baseline,
                    path_results=Path(tmpdir),
                    cfg=cfg
                )
                self.assertTrue(len(os.listdir(tmpdir)) != 0)
