from typing import Type, Union, List, Dict

import torch
from ray.rllib import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.models import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from src.common.observation_space import EDGE_INDEX, EDGE_MASK
from src.nri.utils import get_priors, fully_connected_edge_index, get_prior_tensor
from src.ra_agents.RAFeatureExtractor import RLlibRAGNNModel


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
        total_loss = super().loss(model, dist_class, train_batch)

        # build dynamic prior tensor
        # iterate over batches to build respective prior tensors depending on edges
        all_graph_edges = train_batch["obs"][EDGE_INDEX]
        batched_priors = []
        for batch_index in range(all_graph_edges.shape[0]):
            graph_edges_batch = all_graph_edges[batch_index, :, train_batch["obs"][EDGE_MASK][batch_index]]
            N, E = self.observation_space.num_nodes, graph_edges_batch.shape[1]
            all_edges = fully_connected_edge_index(N)
            prior_for_graph_edges, prior_for_non_graph_edges = get_priors(
                prob_graph_edges_exist=self.config["relation_awareness"]["prior_prob_for_graph_edge"],
                num_graph_edges=E,
                num_non_graph_edges=all_edges.shape[1] - E,
                temperature=self.config["relation_awareness"]["temperature"]
            )
            prior_tensor = get_prior_tensor(
                graph_edges=graph_edges_batch,
                all_edges=all_edges,
                prior_for_graph_edges=prior_for_graph_edges,
                prior_for_non_graph_edges=prior_for_non_graph_edges,
                num_edge_types=self.config["model"]["custom_model_config"]["encoder"]["num_edge_types"]
            )
            batched_priors.append(prior_tensor.to(device=self.device, dtype=torch.float32))

        eps = 0.0000001
        posteriors = model.get_posterior()
        prior_tensor = torch.stack(batched_priors, dim=0)
        kl_loss = (posteriors * (torch.log(posteriors + eps) - torch.log(prior_tensor + eps))).sum(dim=-1)
        kl_loss = kl_loss.mean()

        total_loss += kl_loss * self.config["relation_awareness"]["beta"]

        model.tower_stats["kl_loss"] = kl_loss
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_prior"] = torch.mean(prior_tensor, dim=0) # mean over batch dimension -> [E, K]
        model.tower_stats["mean_posterior"] = torch.mean(posteriors, dim=0) # mean over batch dimension -> [E, K]

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
            "relation_awareness/prior_existence_probs": torch.mean(
                torch.stack([t.tower_stats["mean_prior"][:, 0].detach() for t in self.model_gpu_towers]), dim=0
            ).cpu().numpy().flatten().tolist(),
            "relation_awareness/posterior_existence_probs": torch.mean(
                torch.stack([t.tower_stats["mean_posterior"][:, 0].detach() for t in self.model_gpu_towers]), dim=0
            ).cpu().numpy().flatten().tolist(),
        })

        return stats
