from typing import Type, Union, List

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
        all_graph_edges = train_batch["obs"][EDGE_INDEX] # TODO only zeros
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
                prior_for_non_graph_edges=prior_for_non_graph_edges
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

        return total_loss
