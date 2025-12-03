import unittest

import grid2op
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from stable_baselines3 import PPO

from common.env import G2OpGymEnv
from common.graph_structured_observation_space import BusConnectivityGraphObsSpace
from nri.utils import prior_from_env
from ra_agents.RAFeatureExtractor import RAFeatureExtractorSB3
from ra_agents.ppo import Sb3PPOTopologyPolicy, RAPPO


class TestPPOTopoPolicy(unittest.TestCase):
    def setUp(self):
        self.env = grid2op.make("l2rpn_case14_sandbox")
        self.gym_env = G2OpGymEnv(obs_space_creation=lambda e: BusConnectivityGraphObsSpace(e))
        policy_kwargs = {
            "net_arch": [100, 100, 100],
            "features_extractor_class": RAFeatureExtractorSB3,
            "features_extractor_kwargs": {
                "hidden_dim": 8,
                "out_dim": 8,
                "num_edge_types": 2,
                "num_layers": 2,
                "dropout_prob": 0.0,
                "use_graphormer": True,
                "max_degree": 7,
                "max_path_distance": 9,
            }
        }
        self.ppo = RAPPO(
            env=self.gym_env,
            prior=prior_from_env(0.9, self.gym_env),
            policy_kwargs=policy_kwargs,
        )
        self.topo_policy=Sb3PPOTopologyPolicy(self.ppo)

    def test_forward(self):
        actions = self.topo_policy.get_k_best_actions(self.env.current_obs, 10)
        self.assertEqual(len(actions), 10)