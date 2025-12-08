from typing import List

import numpy as np
import torch
from grid2op.Action import TopologySetAction
from grid2op.Observation import BaseObservation
from stable_baselines3 import DQN

from src.common.baseline_agent import TopologyPolicy
from .RADQN import RAQNetwork


class Sb3DQNTopologyPolicy(TopologyPolicy):
    """
    Implements the interface of a topology policy. Uses a DQN trained by the stable-baselines3 library.
    """

    def __init__(self, alg: DQN):
        """
        Initializes the topology policy that can be used to search for promising moves

        :param alg: The sb3 DQN
        """
        TopologyPolicy.__init__(self)
        self.dqn: DQN = alg
        if not hasattr(alg.observation_space, "to_gym") or not hasattr(alg.action_space, "from_gym"):
            raise ValueError("In order to use your DQN in a grid2op topo policy the DQN observation space needs a"
                             "method 'to_gym' and the action space needs a method 'from_gym'")

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        # Convert observation to a tensor
        gym_obs = self.dqn.observation_space.to_gym(observation)
        obs_tensor, _ = self.dqn.policy.obs_to_tensor(gym_obs)

        # Get Q-values for each action
        with torch.no_grad():
            if isinstance(self.dqn.policy.q_net, RAQNetwork):
                q_values, _ = self.dqn.policy.q_net.forward(obs_tensor)
            else:
                q_values = self.dqn.policy.q_net.forward(obs_tensor)
            q_values = q_values.squeeze().cpu().numpy()
        # Get the indices of the top k actions based on their Q-values
        top_k_indices = np.argsort(q_values)[-k:]  # Sort in descending order
        top_k_actions = [self.dqn.action_space.from_gym(idx) for idx in top_k_indices]

        return top_k_actions