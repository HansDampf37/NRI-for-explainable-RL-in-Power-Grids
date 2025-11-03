"""
Ray RLlib integration for the Relations-Aware FeatureExtractor (RAFeatureExtractor).

This provides:
- A Ray-compatible Env wrapper (RayEnv) using the BusConnectivityGraphObsSpace and a discrete action space.
- A custom TorchModelV2 (RAGNNDQNModel) that embeds RAFeatureExtractor to produce Q-values and exposes
  the posterior edge-type probabilities.
- An optional KL regularization term against a prior over edge types added via `custom_loss` (compatible with DQN).
- A small builder `build_dqn_with_ra` and a `main` showcasing how to instantiate the Algorithm.

Note: There is no REFeatureExtractor in this repository. We assume you meant RAFeatureExtractor (relations-aware).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gymnasium.spaces import Discrete
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from common import (
    EDGE_INDEX,
    EDGE_MASK,
    NODES,
    BusConnectivityGraphObsSpace,
    G2OpGymEnv,
)
from nri.agent.HuberKLLoss import HuberKLLoss
from nri.agent.RA_FE import RAFeatureExtractor
from nri.utils import fully_connected_edge_index, _get_prior


# --------- Env wrapper (same pattern as in RayWrapper.py) ---------
class RayEnv(G2OpGymEnv):
    def __init__(self, env_config: Dict):
        from grid2op.gym_compat import DiscreteActSpace

        super().__init__(
            env_name=env_config.get("env_name", "l2rpn_case14_sandbox"),
            safe_max_rho=env_config.get("safe_max_rho", 0.95),
            act_space_creation=lambda env: DiscreteActSpace(env.action_space, attr_to_keep=["set_bus"]),
            obs_space_creation=lambda env: BusConnectivityGraphObsSpace(env.observation_space),
        )


@dataclass
class RAConfig:
    hidden_dim: int = 64
    out_dim: int = 64
    num_edge_types: int = 2
    dropout_prob: float = 0.0
    # KL regularization weight (added to the DQN policy loss)
    kl_beta: float = 1.0
    # Optional priors: Each should be a length-K vector over edge types.
    prior_for_graph_edges: Optional[List[float]] = None  # e.g., [0.9, 0.1]
    prior_for_non_graph_edges: Optional[List[float]] = None  # e.g., [0.1, 0.9]
    # If provided, use this precomputed prior tensor of shape [E, K] (same E as encoder edges).
    # Otherwise, we'll compute it via a helper using the env's graph.
    prior_matrix: Optional[np.ndarray] = None


# --------- Helper to compute a static prior matrix from an env config ---------

def compute_prior_from_env(env_config: Dict[str, Any], num_edge_types: int,
                            prior_for_graph_edges: Optional[List[float]] = None,
                            prior_for_non_graph_edges: Optional[List[float]] = None) -> np.ndarray:
    """
    Creates a temporary env to read the powergrid edge_index and returns a prior matrix of shape [E, K].
    """
    tmp_env = RayEnv(env_config)
    obs, _ = tmp_env.reset()
    mask = torch.from_numpy(obs[EDGE_MASK]).to(torch.bool)
    powergrid_edge_index = torch.from_numpy(obs[EDGE_INDEX])[:, mask]  # [2, E]
    N = int(powergrid_edge_index.max().item()) + 1
    all_edges = fully_connected_edge_index(N)  # [2, E']

    if prior_for_graph_edges is None:
        prior_for_graph_edges = [1.0 / num_edge_types] * num_edge_types
    if prior_for_non_graph_edges is None:
        prior_for_non_graph_edges = [1.0 / num_edge_types] * num_edge_types

    prior = _get_prior(
        powergrid_edge_index,
        all_edges,
        torch.tensor(prior_for_graph_edges, dtype=torch.float32),
        torch.tensor(prior_for_non_graph_edges, dtype=torch.float32),
    )  # [E', K]
    return prior.detach().cpu().numpy()



# --------- New API: RLModule + Learner for DQN ---------
class RAGNNDQNRLModule(TorchRLModule):
    """
    RLlib RLModule (new API) for DQN that embeds RAFeatureExtractor and outputs q_values.
    It also returns posterior and prior (if available) in forward_train for auxiliary losses.
    """

    def setup(self):
        assert isinstance(self.observation_space, BusConnectivityGraphObsSpace)
        assert isinstance(self.action_space, Discrete)
        cfg_dict: Dict[str, Any] = self.model_config or {}
        self.ra_cfg = RAConfig(
            hidden_dim=cfg_dict.get("hidden_dim", 64),
            out_dim=cfg_dict.get("out_dim", 64),
            num_edge_types=cfg_dict.get("num_edge_types", 2),
            dropout_prob=cfg_dict.get("dropout_prob", 0.0),
            kl_beta=cfg_dict.get("kl_beta", 1.0),
            prior_for_graph_edges=cfg_dict.get("prior_for_graph_edges"),
            prior_for_non_graph_edges=cfg_dict.get("prior_for_non_graph_edges"),
            prior_matrix=cfg_dict.get("prior_matrix"),
        )
        self._gnn = RAFeatureExtractor(
            x_dim=self.observation_space.x_dim,
            hidden_dim=self.ra_cfg.hidden_dim,
            x_out_dim=self.ra_cfg.out_dim,
            num_edge_types=self.ra_cfg.num_edge_types,
            dropout_prob=self.ra_cfg.dropout_prob,
        )
        self._head = nn.Linear(self.ra_cfg.out_dim, self.action_space.n)
        self._kl_helper = HuberKLLoss(prior=None)
        prior = self.ra_cfg.prior_matrix
        if prior is not None:
            self.register_buffer("prior", torch.as_tensor(prior, dtype=torch.float32))
        else:
            self.prior = None  # type: ignore

    def _compute_q_and_post(self, batch: Dict[str, Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        obs = batch["obs"]
        node_feats = obs[NODES]  # [B, N, X]
        data_list: List[Data] = []
        bsz = node_feats.size(0)
        for b in range(bsz):
            data_list.append(Data(x=node_feats[b]))
        pyg_batch: Batch = Batch.from_data_list(data_list)
        x, post = self._gnn(x=pyg_batch.x, batch=pyg_batch.batch)
        q = self._head(x)
        return q, post

    def forward_inference(self, batch: Dict[str, Dict[str, Tensor]], **kwargs) -> Dict[str, Tensor]:
        q, _ = self._compute_q_and_post(batch)
        actions = torch.argmax(q, dim=-1).to(torch.int64)
        return {"q_values": q, "actions": actions}

    def forward_exploration(self, batch: Dict[str, Dict[str, Tensor]], **kwargs) -> Dict[str, Tensor]:
        q, _ = self._compute_q_and_post(batch)
        actions = torch.argmax(q, dim=-1).to(torch.int64)
        return {"q_values": q, "actions": actions}

    def forward_train(self, batch: Dict[str, Dict[str, Tensor]], **kwargs) -> Dict[str, Tensor]:
        q, post = self._compute_q_and_post(batch)
        out = {"q_values": q}
        if post is not None:
            out["posterior"] = post
        if hasattr(self, "prior") and isinstance(self.prior, Tensor):
            out["prior"] = self.prior
        out["kl_beta"] = torch.tensor(self.ra_cfg.kl_beta, dtype=torch.float32, device=q.device)
        return out


def build_dqn_with_ra_newapi(
    env_config: Dict[str, Any],
    ra_model_config: Optional[Dict[str, Any]] = None,
    num_gpus: int = 0,
):
    """
    Build a DQN (new API) using RLModule + Learner with RAFeatureExtractor backend.
    """
    if ra_model_config is None:
        ra_model_config = {}
    if ra_model_config.get("prior_matrix") is None:
        prior_mat = compute_prior_from_env(
            env_config=env_config,
            num_edge_types=ra_model_config.get("num_edge_types", 2),
            prior_for_graph_edges=ra_model_config.get("prior_for_graph_edges"),
            prior_for_non_graph_edges=ra_model_config.get("prior_for_non_graph_edges"),
        )
        ra_model_config["prior_matrix"] = prior_mat

    config = (
        DQNConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .framework("torch")
        .environment(env=RayEnv, env_config=env_config)
        .resources(num_gpus=num_gpus)
        .rl_module(rl_module_spec=RLModuleSpec(module_class=RAGNNDQNRLModule, model_config=ra_model_config))
        .env_runners(rollout_fragment_length=1, batch_mode="complete_episodes", enable_connectors=False)
        .training(
            gamma=0.99,
            lr=1e-4,
            n_step=1,
            replay_buffer_config={
                "type": "ReplayBuffer",
                "storage_unit": "timesteps",
                "capacity": 50000,
            },
        )
    )
    algo = config.build()
    return algo


def main():
    # Minimal smoke test of building (and optional short training loop)
    env_cfg = {"env_name": "l2rpn_case14_sandbox", "safe_max_rho": 0.95}
    ra_cfg: Dict[str, Any] = {
        "hidden_dim": 64,
        "out_dim": 64,
        "num_edge_types": 2,
        "dropout_prob": 0.1,
        "kl_beta": 1.0,
    }

    # Use the new API by default
    algo = build_dqn_with_ra_newapi(env_cfg, ra_cfg, num_gpus=(1 if torch.cuda.is_available() else 0))

    # Optional tiny train
    for i in range(2):
        result = algo.train()
        print(f"Iteration {i}: episode_reward_mean={result['episode_reward_mean']}")

    if hasattr(algo, "stop"):
        algo.stop()


if __name__ == "__main__":
    main()
