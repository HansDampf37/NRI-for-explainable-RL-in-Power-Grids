from typing import List

import numpy as np
import torch
from grid2op.Action import TopologySetAction
from grid2op.Observation import BaseObservation
from stable_baselines3 import PPO

from src.common.baseline_agent import TopologyPolicy


class Sb3PPOTopologyPolicy(TopologyPolicy):
    """
    Implements the interface of a topology policy. Uses a PPO trained by the stable-baselines3 library.
    """

    def __init__(self, alg: PPO):
        """
        Initializes the topology policy that can be used to search for promising moves

        :param alg: The sb3 DQN
        """
        TopologyPolicy.__init__(self)
        self.ppo: PPO = alg
        if not hasattr(alg.observation_space, "to_gym") or not hasattr(alg.action_space, "from_gym"):
            raise ValueError("In order to use your PPO in a grid2op topo policy the PPO observation space needs a"
                             "method 'to_gym' and the action space needs a method 'from_gym'")

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        # Convert observation to a tensor
        gym_obs = self.ppo.observation_space.to_gym(observation)
        obs_tensor, _ = self.ppo.policy.obs_to_tensor(gym_obs)

        # Get action probs for each action
        with torch.no_grad():
            action_probs = self.ppo.policy.get_distribution(obs_tensor).distribution.probs
            action_probs = action_probs.squeeze().cpu().numpy()

        # Get the indices of the top k actions based on their probabilities
        top_k_indices = np.argsort(action_probs)[-k:]  # Sort and pick best k
        top_k_actions = [self.ppo.action_space.from_gym(idx) for idx in top_k_indices] # transform to grid2op format

        return top_k_actions