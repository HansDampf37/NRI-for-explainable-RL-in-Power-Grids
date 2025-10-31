from typing import Dict

import torch
from grid2op.gym_compat import DiscreteActSpace
from gymnasium.spaces import Discrete
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.core.rl_module import RLModule, RLModuleSpec
from torch import nn
from torch_geometric.data import Data, Batch

from common import GraphObservationSpace, NODES, Grid2OpEnvWrapper, BusConnectivityGraphObsSpace
from nri.agent.NRI import NRI_GNN


class RayEnv(Grid2OpEnvWrapper):
    def __init__(self, env_config: Dict):
        super().__init__(
            env_name=env_config["env_name"],
            safe_max_rho=env_config["safe_max_rho"],
            act_space_creation=lambda env: DiscreteActSpace(env.action_space),
            obs_space_creation=lambda env: BusConnectivityGraphObsSpace(env.observation_space)
        )


class NRI_DQN(RLModule, nn.Module):
    def __init__(self, observation_space: GraphObservationSpace, action_space: Discrete, model_config: Dict, catalog_class, inference_only: bool = False, learner_only: bool = False):
        RLModule.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class)
        nn.Module.__init__(self)

        self.nri_gnn = NRI_GNN(
            x_dim=observation_space.x_dim,
            hidden_dim=model_config["hidden_dim"],
            x_out_dim=model_config["hidden_dim"],
            num_edge_types=model_config["num_edge_types"],
            dropout_prob=model_config["dropout_prob"]
        )
        self.linear = nn.Linear(model_config["hidden_dim"], action_space.n)

    def forward(self, input_dict: dict[str, torch.Tensor], state, seq_lens):
        observations = input_dict["obs"]
        node_features_batch = observations[NODES]  # [B, N, node_in_dim]

        data_list = []
        batch_size = node_features_batch.size(0)

        for b in range(batch_size):
            node_features = node_features_batch[b]
            data_list.append(Data(x=node_features))

        batch: Batch = Batch.from_data_list(data_list)
        x, p_z_given_x = self.nri_gnn.forward(x=batch.x, batch=batch.batch)
        x = self.linear(x)
        return x, p_z_given_x


def main():
    config = (
        DQNConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .environment(
            env=RayEnv,
            env_config={
                "env_name": "l2rpn_case14_sandbox",
                "safe_max_rho": 0.95
            }
        )
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
    algo = DQN(config=config)


if __name__ == "__main__":
    main()
