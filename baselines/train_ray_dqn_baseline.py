from pathlib import Path
from pprint import pprint
from typing import List

import numpy as np
import torch
from grid2op.Action import TopologySetAction
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import GymnasiumActionSpace, GymnasiumObservationSpace
from ray.rllib.algorithms import DQNConfig
from ray.rllib.core.rl_module import RLModule

from baseline_gnn_agent.baseline_agent import TopologyPolicy
from common import Grid2OpEnvWrapper

class TopoPolicyDQN(TopologyPolicy[GymnasiumActionSpace, GymnasiumObservationSpace]):
    def __init__(self, action_space: GymnasiumActionSpace, observation_space: GymnasiumObservationSpace, model_checkpoint: Path):
        TopologyPolicy.__init__(self, action_space, observation_space)
        # Load the RL module from the checkpoint
        self.rl_module = RLModule.from_checkpoint(
            model_checkpoint
            / "learner_group"
            / "learner"
            / "rl_module"
            / "default_policy"
        )

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        # Convert observation to a tensor
        obs_batch = torch.from_numpy(self.observation_space.to_gym(observation)).unsqueeze(0)
        # Forward pass to get Q-values (assuming 'forward_inference' returns a tensor of Q-values)
        model_outputs = self.rl_module.forward_inference({"obs": obs_batch})
        q_values = model_outputs["q_values"].squeeze().cpu().detach().numpy()  # Get Q-values for each action
        # Get the indices of the top k actions based on their Q-values
        top_k_indices = np.argsort(q_values)[-k:][::-1]  # Sort in descending order
        top_k_actions = [self.action_space.from_gym(idx) for idx in top_k_indices]

        return top_k_actions


def main():
    config = (
        DQNConfig()
        .environment(Grid2OpEnvWrapper)
        .framework("torch")
        .training(
            dueling=True,
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 60000,
                "alpha": 0.5,
                "beta": 0.5,
            }
        )
        .env_runners(num_env_runners=1)
        .evaluation(evaluation_interval=10)
    )

    algo = config.build_algo()
    for _ in range(100):
        pprint(algo.train())


if __name__ == '__main__':
    main()