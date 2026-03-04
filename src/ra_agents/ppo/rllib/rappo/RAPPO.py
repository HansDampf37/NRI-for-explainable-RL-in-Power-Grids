from typing import Type, Union, List, Dict

import numpy as np
import torch
from ray.rllib import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.models import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from src.common.observation_space import EDGE_INDEX, EDGE_MASK
from src.nri.utils import get_priors, fully_connected_edge_index, get_prior_tensor
from src.ra_agents.RAFeatureExtractor import RLlibRAGNNModel
from ray.rllib.algorithms.registry import POLICIES


class RAPPOTorchPolicy(PPOTorchPolicy):
    @override(PPOTorchPolicy)
    def loss(
            self,
            model: RLlibRAGNNModel,
            dist_class: Type[ActionDistribution],
            train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        # Initialize annealed parameters on first call (lazy initialization)
        ra_config = self.config["relation_awareness"]
        sampling_config = self.config["model"]["custom_model_config"]["sampling"]
        if not hasattr(self, 'current_beta'):
            self.current_beta = ra_config["beta_end"]
            self.target_beta = ra_config["beta_end"]
            self.current_beta_non_graph_edges = ra_config.get("beta_non_graph_edges_end", ra_config["beta_end"])
            self.target_beta_non_graph_edges = ra_config.get("beta_non_graph_edges_end", ra_config["beta_end"])
            self.current_tau = sampling_config["tau_end"]
            self.target_tau = sampling_config["tau_end"]

        total_loss = super().loss(model, dist_class, train_batch)

        # build dynamic prior tensor
        # iterate over batches to build respective prior tensors depending on edges
        all_graph_edges = train_batch["obs"][EDGE_INDEX]
        batched_priors = []
        batched_graph_edge_masks = []

        for batch_index in range(all_graph_edges.shape[0]):
            graph_edges_batch = all_graph_edges[batch_index, :, train_batch["obs"][EDGE_MASK][batch_index]]
            N = self.observation_space.num_nodes if hasattr(self.observation_space, 'num_nodes') else 57
            E = graph_edges_batch.shape[1]
            all_edges = fully_connected_edge_index(N)
            if not ra_config.get("use_per_edge_prior", True):
                prior_for_graph_edges = torch.from_numpy(np.array(ra_config.get("global_prior", [0.9, 0.1])))
                prior_for_non_graph_edges = prior_for_graph_edges # same prior for all edges if not using per-edge priors
            else:
                prior_for_graph_edges, prior_for_non_graph_edges = get_priors(
                    prob_graph_edges_exist=self.config["relation_awareness"]["prior_prob_for_graph_edge"],
                    num_graph_edges=E,
                    num_non_graph_edges=all_edges.shape[1] - E,
                    temperature=self.config["relation_awareness"]["temperature"]
                )
            prior_tensor, graph_edge_mask = get_prior_tensor(
                graph_edges=graph_edges_batch,
                all_edges=all_edges,
                prior_for_graph_edges=prior_for_graph_edges,
                prior_for_non_graph_edges=prior_for_non_graph_edges,
                num_edge_types=self.config["model"]["custom_model_config"]["encoder"]["num_edge_types"],
                return_mask=True,
            )
            batched_priors.append(prior_tensor.to(device=self.device, dtype=torch.float32))
            batched_graph_edge_masks.append(graph_edge_mask.to(device=self.device))

        eps = 0.0000001
        posteriors = model.get_posterior() # [B, E, K]
        prior_tensor = torch.stack(batched_priors, dim=0) # [B, E, K]
        graph_edge_masks = torch.stack(batched_graph_edge_masks, dim=0) # [B, E]

        # Calculate KL divergence for each edge
        kl_per_edge = (posteriors * (torch.log(posteriors + eps) - torch.log(prior_tensor + eps))).sum(dim=-1)  # [B, E]

        # Split KL loss into graph edges and non-graph edges and compute means
        kl_loss_graph_edges = kl_per_edge[graph_edge_masks].mean() if graph_edge_masks.any() else torch.as_tensor(0.0, device=self.device)
        kl_loss_non_graph_edges = kl_per_edge[~graph_edge_masks].mean() if (~graph_edge_masks).any() else torch.as_tensor(0.0, device=self.device)

        # Calculate occurrence fractions
        num_graph_edges = graph_edge_masks.sum().float()
        num_non_graph_edges = (~graph_edge_masks).sum().float()
        total_edges = num_graph_edges + num_non_graph_edges

        fraction_graph_edges = num_graph_edges / total_edges if total_edges > 0 else torch.as_tensor(0.0, device=self.device)
        fraction_non_graph_edges = num_non_graph_edges / total_edges if total_edges > 0 else torch.as_tensor(0.0, device=self.device)

        # Weighted KL loss: each component weighted by occurrence fraction and respective beta
        kl_loss = (fraction_graph_edges * self.current_beta * kl_loss_graph_edges +
                   fraction_non_graph_edges * self.current_beta_non_graph_edges * kl_loss_non_graph_edges)
        kl_div_unweighted = (fraction_graph_edges * kl_loss_graph_edges + fraction_non_graph_edges * kl_loss_non_graph_edges)

        total_loss += kl_loss

        model.tower_stats["kl_loss"] = kl_loss
        model.tower_stats["kl_div_total"] = kl_div_unweighted
        model.tower_stats["kl_div_graph_edges"] = kl_loss_graph_edges
        model.tower_stats["kl_div_non_graph_edges"] = kl_loss_non_graph_edges
        model.tower_stats["fraction_graph_edges"] = fraction_graph_edges
        model.tower_stats["fraction_non_graph_edges"] = fraction_non_graph_edges
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_prior"] = torch.mean(prior_tensor, dim=0) # mean over batch dimension -> [E, K]
        model.tower_stats["mean_posterior"] = torch.mean(posteriors, dim=0) # mean over batch dimension -> [E, K]
        model.tower_stats["current_beta"] = self.current_beta
        model.tower_stats["current_beta_non_graph_edges"] = self.current_beta_non_graph_edges
        model.tower_stats["current_tau"] = self.current_tau
        model.tower_stats["latent_edge_probs"] = posteriors
        model.tower_stats["gnn"] = model.ragnn.gnn.stats

        return total_loss

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Returns a dictionary of stats for TensorBoard."""
        # Get existing stats from PPOTorchPolicy (like policy_loss, vf_loss, etc.)
        stats = super().stats_fn(train_batch)

        # Access the tower_stats we populated in the loss function
        # Note: RLlib averages tower_stats across GPUs automatically
        stats.update({
            "relation_awareness/kl_loss": torch.mean(
                torch.stack([t.tower_stats["kl_loss"].detach() for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/kl_div_total": torch.mean(
                torch.stack([t.tower_stats["kl_div_total"].detach() for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/kl_div_graph_edges": torch.mean(
                torch.stack([t.tower_stats["kl_div_graph_edges"].detach() for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/kl_div_non_graph_edges": torch.mean(
                torch.stack([t.tower_stats["kl_div_non_graph_edges"].detach() for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/current_beta": torch.mean(
                torch.stack([torch.as_tensor(t.tower_stats["current_beta"], dtype=torch.float32, device=self.device) for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/target_beta": float(self.target_beta),
            "relation_awareness/current_beta_non_graph_edges": torch.mean(
                torch.stack([torch.as_tensor(t.tower_stats["current_beta_non_graph_edges"], dtype=torch.float32, device=self.device) for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/target_beta_non_graph_edges": float(self.target_beta_non_graph_edges),
            "relation_awareness/fraction_graph_edges": torch.mean(
                torch.stack([t.tower_stats["fraction_graph_edges"].detach() for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/fraction_non_graph_edges": torch.mean(
                torch.stack([t.tower_stats["fraction_non_graph_edges"].detach() for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/current_tau": torch.mean(
                torch.stack([torch.as_tensor(t.tower_stats["current_tau"], dtype=torch.float32, device=self.device) for t in self.model_gpu_towers])
            ).item(),
            "relation_awareness/target_tau":float(self.target_tau),
            "relation_awareness/prior_existence_probs": torch.mean(
                torch.stack([t.tower_stats["mean_prior"][:, 0].detach() for t in self.model_gpu_towers]), dim=0
            ).cpu().numpy().flatten().tolist(),
            "relation_awareness/posterior_existence_probs": torch.mean(
                torch.stack([t.tower_stats["mean_posterior"][:, 0].detach() for t in self.model_gpu_towers]), dim=0
            ).cpu().numpy().flatten().tolist(),
            "relation_awareness/latent_graph_probs_mean": torch.mean(
                torch.cat([t.tower_stats["latent_edge_probs"].detach() for t in self.model_gpu_towers], 0), dim=0
            ).cpu().numpy().tolist(),
            "relation_awareness/latent_graph_probs_var": torch.var(
                torch.cat([t.tower_stats["latent_edge_probs"].detach() for t in self.model_gpu_towers], 0), dim=0
            ).cpu().numpy().tolist(),
        })

        for key, value in self.model_gpu_towers[0].tower_stats["gnn"].items():
            stats[f"relation_awareness/gnn/{key}"] = torch.mean(
                torch.stack([t.tower_stats["gnn"][key].detach() for t in self.model_gpu_towers])
            ).item()

        return stats


POLICIES["rappo_torch_policy"] = RAPPOTorchPolicy