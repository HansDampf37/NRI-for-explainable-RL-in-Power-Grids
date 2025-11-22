import unittest
from pathlib import Path

import grid2op
from tempfile import TemporaryDirectory
from grid2op.Agent import DoNothingAgent
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from stable_baselines3 import DQN

from common.baseline_agent import evaluate_agent, evaluate_sb3_alg

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

    def test_evaluate_sb3_alg(self):
        gym_env = GymEnv(self.env)
        gym_env.action_space = DiscreteActSpace(grid2op_action_space=self.env.action_space)
        dqn = DQN(env=gym_env, policy="MultiInputPolicy")

        with TemporaryDirectory() as tmpdir:
            path_results = Path(tmpdir)
            evaluate_sb3_alg(
                alg=dqn,
                env=gym_env,
                path_results=path_results,
                num_episodes=1,
            )
