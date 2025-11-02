from typing import Dict, Tuple

import ray
import torch
from grid2op.gym_compat import DiscreteActSpace
from gymnasium.spaces import Discrete
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from torch import nn, Tensor
from torch_geometric.data import Data, Batch

from common import NODES, G2OpGymEnv, BusConnectivityGraphObsSpace
from nri.agent.RA_FE import RAFeatureExtractor


class RayEnv(G2OpGymEnv):
    def __init__(self, env_config: Dict):
        super().__init__(
            env_name=env_config["env_name"],
            safe_max_rho=env_config["safe_max_rho"],
            act_space_creation=lambda env: DiscreteActSpace(env.action_space, attr_to_keep=["set_bus"]),
            obs_space_creation=lambda env: BusConnectivityGraphObsSpace(env.observation_space)
        )


class NRI_DQN(TorchRLModule):
    def setup(self):
        assert isinstance(self.observation_space, BusConnectivityGraphObsSpace)
        assert isinstance(self.action_space, Discrete)
        self._nri_gnn = RAFeatureExtractor(
            x_dim=self.observation_space.x_dim,
            hidden_dim=self.model_config["hidden_dim"],
            x_out_dim=self.model_config["hidden_dim"],
            num_edge_types=self.model_config["num_edge_types"],
            dropout_prob=self.model_config["dropout_prob"]
        )
        self._linear = nn.Linear(self.model_config["hidden_dim"], self.action_space.n)


    def _forward(self, batch: dict[str, dict[str, Tensor]], **kwargs) -> Tuple[Tensor, Tensor]:
        observations = batch["obs"]
        node_features_batch = observations[NODES]  # [B, N, node_in_dim]

        data_list = []
        batch_size = node_features_batch.size(0)

        for b in range(batch_size):
            node_features = node_features_batch[b]
            data_list.append(Data(x=node_features))

        batch: Batch = Batch.from_data_list(data_list)
        x, p_z_given_x = self._nri_gnn.forward(x=batch.x, batch=batch.batch)
        x = self._linear(x)
        return x, p_z_given_x


def main():
    ray.init()
    env_config = {
        "env_name": "l2rpn_case14_sandbox",
        "safe_max_rho": 0.95
    }
    config = (
        PPOConfig()
        .training(gamma=0.9, lr=0.01)
        .environment(env=RayEnv, env_config=env_config)
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .env_runners(num_env_runners=0)
        .framework("torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=NRI_DQN,
                model_config={
                    "hidden_dim": 32,
                    "num_edge_types": 2,
                    "dropout_prob": 0.1,
                },
            )
        )
    )

    # A config object can be used to construct the respective Algorithm.
    config.build_algo()
    # config = (
    #     DQNConfig()
    #     .api_stack(enable_rl_module_and_learner=True)
    #     .framework("torch")
    #     .environment(
    #         env=RayEnv,
    #         env_config={
    #             "env_name": "l2rpn_case14_sandbox",
    #             "safe_max_rho": 0.95
    #         }
    #     )
    #     .rl_module(
    #         rl_module_spec=RLModuleSpec(
    #             module_class=NRI_DQN,
    #             model_config={
    #                 "hidden_dim": 32,
    #                 "num_edge_types": 2,
    #                 "dropout_prob": 0.1,
    #             },
    #         )
    #     )
    #     .training(
    #         replay_buffer_config={
    #             "type": "PrioritizedEpisodeReplayBuffer",
    #             "capacity": 60000,
    #             "alpha": 0.5,
    #             "beta": 0.5,
    #         },
    #         n_step=120_000,
    #     )
    #     .env_runners(num_env_runners=0)
    # )
    # algo = DQN(config=config)
    # algo.train()


if __name__ == "__main__":
    main()
