from typing import List, Dict

import numpy as np
import torch
from grid2op.Action import TopologySetAction
from grid2op.Observation import BaseObservation
from stable_baselines3 import DQN, PPO

from baselines.baseline_agent import TopologyPolicy


class TopoPolicyStableDQN(TopologyPolicy):
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
                             "method 'to_gym' and the action space needs a method 'from_gym'")  # TODO bad design

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        # Convert observation to a tensor
        gym_obs = self.dqn.observation_space.to_gym(observation)
        if isinstance(gym_obs, np.ndarray):
            obs_batch = torch.from_numpy(gym_obs).unsqueeze(0).to(dtype=torch.float32, device=self.dqn.device)
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
        top_k_indices = np.argsort(q_values)[-k:]  # Sort in descending order
        top_k_actions = [self.dqn.action_space.from_gym(idx) for idx in top_k_indices]

        return top_k_actions

class TopoPolicyStablePPO(TopologyPolicy):
    """
    Implements the interface of a topology policy. Uses a PPO trained by the stable-baselines3 library.
    """

    def __init__(self, alg: DQN):
        """
        Initializes the topology policy that can be used to search for promising moves

        :param alg: The sb3 PPO
        """
        TopologyPolicy.__init__(self)
        self.ppo: PPO = alg
        if not hasattr(alg.observation_space, "to_gym") or not hasattr(alg.action_space, "from_gym"):
            raise ValueError("In order to use your PPO in a grid2op topo policy the PPO observation space needs a"
                             "method 'to_gym' and the action space needs a method 'from_gym'")

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        # Convert observation to a tensor
        gym_obs = self.ppo.observation_space.to_gym(observation)
        if isinstance(gym_obs, np.ndarray):
            obs_batch = torch.from_numpy(gym_obs).unsqueeze(0).to(dtype=torch.float32, device=self.ppo.device)
        elif isinstance(gym_obs, Dict):
            obs_batch = {}
            for key, value in gym_obs.items():
                obs_batch[key] = torch.from_numpy(value).unsqueeze(0).to(dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unknown gym obs type {gym_obs.__class__.__name__}")

        # Get action probs for each action
        with torch.no_grad():
            action_probs = self.ppo.policy.get_distribution(obs_batch).distribution.probs
            action_probs = action_probs.squeeze()
        # Get the indices of the top k actions based on their Q-values
        top_k_indices = np.argsort(action_probs)[-k:]  # Sort in descending order
        top_k_actions = [self.ppo.action_space.from_gym(idx) for idx in top_k_indices]
        return top_k_actions

