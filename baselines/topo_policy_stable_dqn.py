from typing import List, Dict

import numpy as np
import torch
from grid2op.Action import TopologySetAction
from grid2op.Observation import BaseObservation
from stable_baselines3 import DQN

from baselines.baseline_agent import TopologyPolicy


class TopoPolicyStableDQN(TopologyPolicy):
    """
    Implements the interface of a topology policy. Uses a DQN trained by the stable-baselines3 library.
    """

    def __init__(self, dqn: DQN):
        """
        Initializes the topology policy that can be used to search for promising moves

        :param dqn: The sb3 DQN
        """
        TopologyPolicy.__init__(self)
        self.dqn: DQN = dqn
        if not hasattr(dqn.observation_space, "to_gym") or not hasattr(dqn.action_space, "from_gym"):
            raise ValueError("In order to use your DQN in a grid2op topo policy the DQN observation space needs a"
                             "method 'to_gym' and the action space needs a method 'from_gym'")  # TODO bad design

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        # Convert observation to a tensor
        gym_obs = self.dqn.observation_space.to_gym(observation)
        if isinstance(gym_obs, np.ndarray):
            obs_batch = torch.from_numpy(gym_obs).unsqueeze(0).to(dtype=torch.float32)
        elif isinstance(gym_obs, Dict):
            obs_batch = {}
            for key, value in gym_obs.items():
                obs_batch[key] = torch.from_numpy(value).unsqueeze(0).to(dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unknown gym obs type {gym_obs.__class__.__name__}")

        # Get Q-values for each action
        with torch.no_grad():
            features = self.dqn.policy.q_net.features_extractor(obs_batch)
            q_values = self.dqn.policy.q_net.q_net(features)
            q_values = q_values.squeeze().cpu().numpy()
        # Get the indices of the top k actions based on their Q-values
        top_k_indices = np.argsort(q_values)[-k:][::-1]  # Sort in descending order
        top_k_actions = [self.dqn.action_space.from_gym(idx) for idx in top_k_indices]

        return top_k_actions
