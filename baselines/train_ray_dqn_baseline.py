from pathlib import Path
from pprint import pprint
from typing import List

from grid2op.Action import TopologySetAction
from grid2op.Observation import BaseObservation
from ray.rllib.algorithms import DQNConfig
from ray.rllib.core.rl_module import RLModule

from baselines.baseline_agent import TopologyPolicy
from common import Grid2OpEnvWrapper


class TopoPolicyDQN(TopologyPolicy):
    def __init__(self, model_checkpoint: Path):
        TopologyPolicy.__init__(self)
        # Load the RL module from the checkpoint
        self.rl_module = RLModule.from_checkpoint(
            model_checkpoint
            / "learner_group"
            / "learner"
            / "rl_module"
            / "default_policy"
        )

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        raise NotImplementedError("Not yet implemented")


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
    )

    algo = config.build_algo()
    for _ in range(100):
        pprint(algo.train())


if __name__ == '__main__':
    main()