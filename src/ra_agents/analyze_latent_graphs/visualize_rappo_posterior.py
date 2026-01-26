"""
Simplified script to run a saved RAPPO checkpoint and display the posterior distribution.

This version shows posterior statistics and graph visualizations.
Reuses existing infrastructure from evaluate_rllib_agent.py
"""
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from matplotlib.ticker import MultipleLocator
from ray.rllib.models import ModelCatalog
from torch import Tensor

from evaluate_rllib_agent import load_config, load_rllib_agent
from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX, EDGE_MASK, NODES
from src.nri.utils import fully_connected_edge_index, get_priors, get_prior_tensor
from src.ra_agents.RAFeatureExtractor import RLlibGNNModel, RLlibRAGNNModel, RLlibNRIGNNModel
from src.ra_agents.analyze_latent_graphs.Metrics import (
    DegreeDistributionVisualizer,
    MetricVisualizer,
    PosteriorDistributionVisualizer,
    KLDivergenceVisualizer,
    EntropyVsKLVisualizer,
    PathLengthVisualizer,
    ClusteringCoefficientVisualizer,
    InnerTreeNodeProbabilityVisualizer,
    SymmetryMetricVisualizer,
    EdgeNodeTypeVisualizer
)
from src.rl4pnc.evaluation.evaluation_agents import RllibAgent
from src.rl4pnc.grid2op_env.observation_converter import ObservationConverter
from src.visualization import get_node_styles
from src.visualization import visualize_graph, PlottingArgs
from src.visualization.utils import NodeStyle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PosteriorAnalyser:
    """Class to analyse and visualize RAPPO posterior distributions."""

    def __init__(self, save_dir: Path, node_styles: List[NodeStyle], prior_for_graph_edges: float = 0.9, temperature: float = 0.5, num_edge_types: int = 2, num_posterior_samples: int = 100):
        self.save_dir = save_dir

        self.prior_for_graph_edges = prior_for_graph_edges
        self.temperature = temperature
        self.num_edge_types = num_edge_types
        num_nodes = len(node_styles)
        self.fully_connected_edge_index = fully_connected_edge_index(num_nodes).detach().cpu().numpy()
        self.num_nodes = num_nodes
        self.num_posterior_samples = num_posterior_samples

        self.metrics: Dict[str, MetricVisualizer] = {
            "Node Degree": DegreeDistributionVisualizer(node_styles=node_styles),
            "Clustering Coefficient": ClusteringCoefficientVisualizer(node_styles=node_styles),
            "Inner Tree Node Probability": InnerTreeNodeProbabilityVisualizer(node_styles=node_styles),
            "Posterior Distribution": PosteriorDistributionVisualizer(node_styles=node_styles),
            "KL Divergence": KLDivergenceVisualizer(node_styles=node_styles),
            "Entropy vs KL": EntropyVsKLVisualizer(node_styles=node_styles),
            "Path Length": PathLengthVisualizer(node_styles=node_styles),
            "Symmetry Analysis": SymmetryMetricVisualizer(node_styles=node_styles),
            "Connected Node Types": EdgeNodeTypeVisualizer(node_styles=node_styles),
        }

        self.all_posteriors = []
        self.all_priors = []
        self.all_kls = []
        self.current_chronic_id = None
        self.step_count_this_episode = 0
        self.step_count_total = 0
        self.episode_count = 0

    def on_new_step(self, posterior: npt.NDArray, powergrid_graph: npt.NDArray, observation: BaseObservation):
        self.step_count_this_episode += 1
        self.step_count_total += 1
        prior = self._get_prior(powergrid_graph, self.prior_for_graph_edges, self.temperature, self.num_edge_types)
        self.all_posteriors.append(posterior)
        self.all_priors.append(prior)

        # compute and log all metrics
        # create n samples
        samples = self._sample_n_graphs(posterior, n=self.num_posterior_samples)
        node_mask = self._compute_mask_connected_nodes(posterior, threshold=0.5)
        for metric_name, metric_vis_fn in self.metrics.items():
            save_dir = self.save_dir / "metrics" / metric_name.replace(" ", "_").lower()
            image_save_path = save_dir / f"{self.current_chronic_id}_step_{self.step_count_total:04d}.svg"
            try:
                figure, data = metric_vis_fn(
                    posterior=posterior,
                    prior=prior,
                    samples=samples,
                    powergrid_graph=powergrid_graph,
                    edge_index_fully_connected=self.fully_connected_edge_index,
                    node_mask=node_mask,
                    observation=observation,
                    show_figure=False,
                )
                if figure is None:
                    continue

                save_dir.mkdir(parents=True, exist_ok=True)
                figure.savefig(image_save_path)
                logger.info(f"Saved metric {metric_name} at step {self.step_count_total} to {image_save_path}")
                plt.close(figure)
            except Exception as e:
                logger.error(f"Error computing metric {metric_name} at step {self.step_count_total}: {e}")
                traceback.print_exc()

    def on_new_episode(self, chronic_id: str):
        self.current_chronic_id = chronic_id
        self.episode_count += 1

    def on_evaluation_end(self):
        for metric_name, metric_fn in self.metrics.items():
            parent_dir = self.save_dir / "metrics_agg"
            path_fig = parent_dir / f"{metric_name.replace(' ', '_').lower()}.svg"
            path_data = parent_dir / f"{metric_name.replace(' ', '_').lower()}.pkl"
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                figure, data = metric_fn.summarize(show_figure=False)
                figure.savefig(path_fig)
                metric_fn.save_data(data, path_data)
                plt.close(figure)
                logger.info(f"Saved aggregate metric {metric_name} to {path_fig} and {path_data}")
            except Exception as e:
                logger.error(f"Error computing aggregate metric {metric_name}: {e}")
                traceback.print_exc()

    def _compute_mask_connected_nodes(self, posterior: npt.NDArray, threshold: float = 0.5) -> npt.NDArray:
        """
        Graphs sampled from the posterior partition the nodes into subgraphs that have sparse connections within and subgraphs that are not connected at all.
        To analyze only the main connected component, we compute a mask over nodes that are connected via edges with existence probability above a certain threshold.
        :param posterior: The posterior distribution as a numpy array of shape (num_edges, num_edge_types)
        :param threshold: The threshold for edge existence probability to consider an edge as present
        :return: a boolean mask of shape [num_nodes] indicating which nodes are considered
        """
        edge_probs = posterior[:, 0:-1].sum(axis=1)  # Sum over all edge types except the last one (non-existence)
        edges_above_threshold = self.fully_connected_edge_index[:, edge_probs >= threshold]

        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(edges_above_threshold.T.tolist())

        # Get the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        node_mask = np.zeros(self.num_nodes, dtype=bool)
        node_mask[list(largest_cc)] = True

        return node_mask


    def _visualize_posterior(self, posterior: npt.NDArray, **kwargs):
        """
        Visualizes the posterior distribution by plotting a histogram of edge existence probabilities and displaying the latent graph structure.
        :param posterior: the posterior distribution as a numpy array of shape (num_edges, num_edge_types)
        :param kwargs: additional keyword arguments
        """
        edge_probs = posterior[:, 0]
        # Update plot
        for ax in self.axes:
            ax.clear()

        # Left: Histogram
        self.axes[0].hist(edge_probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        self.axes[0].axvline(edge_probs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {edge_probs.mean():.3f}')
        self.axes[0].axvline(np.median(edge_probs), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(edge_probs):.3f}')
        self.axes[0].set_xlabel('Edge Existence Probability')
        self.axes[0].set_ylabel('Count')
        self.axes[0].set_title(f'Posterior Distribution (RL Action #{self.rl_action_count})')
        self.axes[0].legend()
        self.axes[0].grid(alpha=0.3)

        gym_obs = self.gym_wrapper.cur_gym_obs
        powerline_edges = gym_obs[EDGE_INDEX][:, gym_obs[EDGE_MASK]]

        # Right: Draw graph directly into subplot using visualize_graph
        self.axes[1].clear()
        visualize_graph(PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=posterior,
            skip_last_edge_type=True
        ), ax=self.axes[1])

        self.axes[1].set_title('Latent Graph Structure\n(Black dashed = power grid, Colored = latent)')

        # Save
        save_dir = self.save_dir / "posteriors"
        save_dir.mkdir(parents=True, exist_ok=True)
        chronic_name = self.chronic_id if self.chronic_id else "unknown"

        self.fig.suptitle(
            f'RAPPO Posterior',
            fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save combined figure
        save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        self.fig.savefig(save_path, dpi=120, bbox_inches='tight')

        print(f"\nSaved to: {save_path}")

    def _sample_n_graphs(self, posterior: npt.NDArray, n: int) -> list[npt.NDArray]:
        """
        Samples n edge_indices from the posterior distribution.
        :param posterior: the posterior distribution as a numpy array of shape (num_edges, num_edge_types)
        :param n: number of graphs to sample
        :return: list of sampled edge indices as numpy arrays
        """
        sampled_graphs = []
        E, K = posterior.shape
        for _ in range(n):
            edge_exists = np.random.random(E) < posterior[:, 0]
            sampled_edges = self.fully_connected_edge_index[:, edge_exists]
            sampled_graphs.append(sampled_edges)

        return sampled_graphs

    def _get_prior(self, powergrid_graph: npt.NDArray, prior_for_graph_edges: float = 0.9, temperature: float = 0.5, num_edge_types: int = 2) -> npt.NDArray:
        """
        Returns the prior distribution over types for each edge as a numpy array
        :param powergrid_graph: the current powergrid graph edge index
        :param prior_for_graph_edges: the prior probability to exist for edges in the powergrid graph
        :param temperature: the fraction of additional edges that we want to infer on top of the powergrid graph
        :param num_edge_types: the number of edge types
        :return: the prior distribution as a numpy array of shape (num_edges, num_edge_types)
        """
        E = powergrid_graph.shape[1]
        prior_for_graph_edges, prior_for_non_graph_edges = get_priors(
            prob_graph_edges_exist=prior_for_graph_edges,
            num_graph_edges=E,
            num_non_graph_edges=self.fully_connected_edge_index.shape[1] - E,
            temperature=temperature
        )
        prior = get_prior_tensor(
            graph_edges=torch.from_numpy(powergrid_graph),
            all_edges=torch.from_numpy(self.fully_connected_edge_index),
            prior_for_graph_edges=prior_for_graph_edges,
            prior_for_non_graph_edges=prior_for_non_graph_edges,
            num_edge_types=num_edge_types,
        ).detach().cpu().numpy()
        return prior


class LatentGraphAnalysisAgent(BaseAgent):
    """Wrapper around RllibAgent that displays RAPPO posterior distribution when RL model acts."""

    def __init__(self, rllib_agent: RllibAgent, gym_wrapper: ObservationConverter, analyser: PosteriorAnalyser):
        """Initialize RAPPOPosteriorAgent."""
        BaseAgent.__init__(self, rllib_agent.action_space)

        self.rllib_agent = rllib_agent
        self.analyser = analyser
        self.gym_wrapper = gym_wrapper

        # Get the policy model for accessing posterior
        self.policy_model = rllib_agent._rllib_agent.model

        # Get environment and node styles for graph visualization
        self.g2op_env = gym_wrapper.env_gym.init_env

    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        """Returns action and visualizes posterior when RL model is used."""
        action = self.rllib_agent.act(observation, reward, done)

        # Check if RL agent will be activated (same logic as RllibAgent)
        use_rl_component = self.rllib_agent.activate_agent(observation)
        if use_rl_component:
            posterior = self._get_posterior()
            self.rllib_agent.gym_wrapper.update_obs(observation)
            powergrid_edge_index = self.rllib_agent.gym_wrapper.cur_gym_obs[EDGE_INDEX]
            edge_mask = self.rllib_agent.gym_wrapper.cur_gym_obs[EDGE_MASK]
            powergrid_edge_index = torch.from_numpy(powergrid_edge_index[..., edge_mask])

            self.analyser.on_new_step(
                posterior,
                powergrid_edge_index.detach().cpu().numpy(),
                observation
            )

        return action

    def on_new_episode(self, chronic_id: str):
        """Notify analyser of new episode."""
        self.analyser.on_new_episode(chronic_id)

    def on_evaluation_end(self):
        """Notify analyser of evaluation end."""
        self.analyser.on_evaluation_end()

    def analyse(self, observation: BaseObservation):
        """Prints info and saves all kinds of plots about the previously computed posterior."""
        posterior = self._get_posterior()
        prior, graph_edge_mask = self._get_prior(observation)

        # Store for later aggregation
        self.all_posteriors.append(posterior.copy())
        self.all_priors.append(prior)
        self.all_edge_masks.append(graph_edge_mask)
        per_edge_kl = self._compute_per_edge_kl(posterior, prior)
        self.all_kls.append(per_edge_kl)

        rho_max = observation.rho.max()
        edge_probs = posterior[:, 0]  # Edge existence probabilities

        self._print_info(edge_probs, posterior, rho_max)
        #self._visualize_posterior(posterior, rho_max)
        #self._visualize_kl(per_edge_kl)
        #self._visualize_entropy_vs_kl(per_edge_kl, posterior, rho_max)
        #self._visualize_graph_properties(posterior)
        #self._visualize_path_length(posterior)

    def _visualize_posterior(self, posterior: npt.NDArray, rho_max: float):
        edge_probs = posterior[:, 0]
        # Update plot
        for ax in self.axes:
            ax.clear()

        # Left: Histogram
        self.axes[0].hist(edge_probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        self.axes[0].axvline(edge_probs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {edge_probs.mean():.3f}')
        self.axes[0].axvline(np.median(edge_probs), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(edge_probs):.3f}')
        self.axes[0].set_xlabel('Edge Existence Probability')
        self.axes[0].set_ylabel('Count')
        self.axes[0].set_title(f'Posterior Distribution (RL Action #{self.rl_action_count})')
        self.axes[0].legend()
        self.axes[0].grid(alpha=0.3)

        gym_obs = self.gym_wrapper.cur_gym_obs
        powerline_edges = gym_obs[EDGE_INDEX][:, gym_obs[EDGE_MASK]]

        # Right: Draw graph directly into subplot using visualize_graph
        self.axes[1].clear()
        visualize_graph(PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=posterior,
            skip_last_edge_type=True
        ), ax=self.axes[1])

        self.axes[1].set_title('Latent Graph Structure\n(Black dashed = power grid, Colored = latent)')

        # Save
        save_dir = self.save_dir / "posteriors"
        save_dir.mkdir(parents=True, exist_ok=True)
        chronic_name = self.chronic_id if self.chronic_id else "unknown"

        self.fig.suptitle(
            f'RAPPO Posterior',
            fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save combined figure
        save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        self.fig.savefig(save_path, dpi=120, bbox_inches='tight')

        print(f"\nSaved to: {save_path}")

    def _compute_per_edge_kl(self, posterior: npt.NDArray, prior: npt.NDArray) -> npt.NDArray:
        eps = 0.0000001
        posterior = torch.from_numpy(posterior)
        prior = torch.from_numpy(prior)
        per_edge_kl = (posterior * (torch.log(posterior + eps) - torch.log(prior + eps))).sum(dim=-1).cpu().detach().numpy()
        return per_edge_kl

    def _get_prior(self, observation: BaseObservation) -> tuple[Tensor, Tensor]:
        self.rllib_agent.gym_wrapper.update_obs(observation)
        edge_index = self.rllib_agent.gym_wrapper.cur_gym_obs[EDGE_INDEX]
        edge_mask = self.rllib_agent.gym_wrapper.cur_gym_obs[EDGE_MASK]
        edge_index = torch.from_numpy(edge_index[..., edge_mask])
        N = self.rllib_agent.gym_wrapper.cur_gym_obs[NODES].shape[0]
        E = edge_index.shape[1]
        all_edges = fully_connected_edge_index(N)
        prior_for_graph_edges, prior_for_non_graph_edges = get_priors(
            prob_graph_edges_exist=0.9,
            num_graph_edges=E,
            num_non_graph_edges=all_edges.shape[1] - E,
            temperature=0.5
        )
        prior_tensor, graph_edge_mask = get_prior_tensor(
            graph_edges=edge_index,
            all_edges=all_edges,
            prior_for_graph_edges=prior_for_graph_edges,
            prior_for_non_graph_edges=prior_for_non_graph_edges,
            num_edge_types=2,
            return_mask=True,
        )
        return prior_tensor.detach().cpu().numpy(), graph_edge_mask.detach().cpu().numpy()

    def _get_posterior(self) -> Any:
        posterior = self.policy_model.get_posterior()  # [B, E, K]

        if posterior.dim() == 3:
            posterior = posterior[0]  # [E, K]

        posterior_np = posterior.cpu().detach().numpy()
        return posterior_np

    def _visualize_kl(self, per_edge_kl: npt.NDArray[np.float32], is_aggregate: bool = False):
        """Plot histogram of per-edge KL divergence values."""
        # Create new figure for aggregate, use existing for per-step
        if is_aggregate:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        else:
            for ax in self.axes:
                ax.clear()
            fig, axes = self.fig, self.axes

        # Left: Histogram
        axes[0].hist(per_edge_kl, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(per_edge_kl.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean KL: {per_edge_kl.mean():.3f}')
        axes[0].axvline(np.median(per_edge_kl), color='green', linestyle='--', linewidth=2, label=f'Median KL: {np.median(per_edge_kl):.3f}')
        axes[0].set_xlabel('Per-Edge KL Divergence')
        axes[0].set_ylabel('Count')
        if is_aggregate:
            axes[0].set_title(f'Histogram of Per-Edge KL Divergence Values\n(Based on mean posterior and prior)')
        else:
            axes[0].set_title(f'Histogram of Per-Edge KL Divergence Values')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        gym_obs = self.gym_wrapper.cur_gym_obs
        powerline_edges = gym_obs[EDGE_INDEX][:, gym_obs[EDGE_MASK]]

        # Right: Draw graph directly into subplot using visualize_graph
        if not is_aggregate:
            axes[1].clear()
        edge_weights = np.stack([per_edge_kl, np.zeros(len(per_edge_kl))]).transpose() / per_edge_kl.max()
        visualize_graph(PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=edge_weights,
            skip_last_edge_type=True
        ), ax=axes[1])

        axes[1].set_title('KL per edge Graph Structure\n(Black dashed = power grid, Colored = KL value)')

        # Save
        save_dir = self.save_dir / "kl"
        save_dir.mkdir(parents=True, exist_ok=True)
        chronic_name = self.chronic_id if self.chronic_id else "unknown"

        if is_aggregate:
            fig.suptitle(
                f'RAPPO KL (Mean Posterior vs Mean Prior) - {chronic_name} ({len(self.all_posteriors)} inferences)',
                fontsize=14, fontweight='bold')
        else:
            fig.suptitle(
                f'RAPPO KL',
                fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save combined figure
        if is_aggregate:
            save_path = save_dir / f"{chronic_name}_mean_kl.png"
        else:
            save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        fig.savefig(save_path, dpi=120, bbox_inches='tight')

        print(f"\nSaved KL to: {save_path}")

        if is_aggregate:
            plt.close(fig)


    def generate_aggregate_figures(self, save_dir: Path):
        """Generate mean and variance figures across all posteriors."""
        if len(self.all_posteriors) == 0:
            logger.warning("No posteriors collected, skipping aggregate figures")
            return

        print(f"\n{'='*80}")
        print(f"Generating aggregate statistics from {len(self.all_posteriors)} posteriors")
        print(f"{'='*80}\n")

        # Stack all posteriors: [N_inferences, E, K]
        all_posteriors_stacked = np.stack(self.all_posteriors, axis=0)
        all_priors_stacked = np.stack(self.all_priors, axis=0)


        # Compute mean and variance across inferences (axis 0)
        mean_posterior = np.mean(all_posteriors_stacked, axis=0)  # [E, K]
        mean_prior = np.mean(all_priors_stacked, axis=0)  # [E, K]
        var_posterior = np.var(all_posteriors_stacked, axis=0)    # [E, K]
        per_edge_kl = self._compute_per_edge_kl(mean_posterior, mean_prior)
        self._visualize_kl(per_edge_kl=per_edge_kl, is_aggregate=True)
        self._visualize_path_length(mean_posterior, is_aggregate=True)
        self._visualize_graph_properties(mean_posterior, is_aggregate=True)

        mean_edge_probs = mean_posterior[:, 0]  # Edge existence probabilities
        var_edge_probs = var_posterior[:, 0]
        std_edge_probs = np.sqrt(var_edge_probs)

        # Print statistics
        print(f"Mean Edge Existence Probability Statistics:")
        print(f"  Mean of means:   {mean_edge_probs.mean():.4f}")
        print(f"  Std of means:    {mean_edge_probs.std():.4f}")
        print(f"  Min of means:    {mean_edge_probs.min():.4f}")
        print(f"  Max of means:    {mean_edge_probs.max():.4f}")
        print(f"\nVariance Statistics:")
        print(f"  Mean variance:   {var_edge_probs.mean():.4f}")
        print(f"  Mean std:        {std_edge_probs.mean():.4f}")
        print(f"  Max variance:    {var_edge_probs.max():.4f}")
        print(f"  Min variance:    {var_edge_probs.min():.4f}")

        # Get power grid edges from last observation
        from src.common.observation_space import EDGE_INDEX, EDGE_MASK
        from src.visualization import visualize_graph, PlottingArgs

        gym_obs = self.gym_wrapper.cur_gym_obs
        powerline_edges = gym_obs[EDGE_INDEX][:, gym_obs[EDGE_MASK]]

        # === FIGURE 1: Mean Posterior ===
        fig_mean, axes_mean = plt.subplots(1, 2, figsize=(20, 8))

        # Left: Histogram of mean edge probabilities
        axes_mean[0].hist(mean_edge_probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes_mean[0].axvline(mean_edge_probs.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_edge_probs.mean():.3f}')
        axes_mean[0].axvline(np.median(mean_edge_probs), color='green', linestyle='--', linewidth=2,
                           label=f'Median: {np.median(mean_edge_probs):.3f}')
        axes_mean[0].set_xlabel('Edge Existence Probability')
        axes_mean[0].set_ylabel('Count')
        axes_mean[0].set_title(f'Mean Posterior Distribution (Averaged over {len(self.all_posteriors)} inferences)')
        axes_mean[0].legend()
        axes_mean[0].grid(alpha=0.3)

        # Right: Graph visualization with mean posteriors
        visualize_graph(PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=mean_posterior,
            skip_last_edge_type=True
        ), ax=axes_mean[1])
        axes_mean[1].set_title('Mean Latent Graph Structure\n(Black dashed = power grid, Colored = mean latent)')

        chronic_name = self.chronic_id if self.chronic_id else "unknown"
        fig_mean.suptitle(f'Mean RAPPO Posterior - {chronic_name} ({len(self.all_posteriors)} inferences)',
                        fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save mean figure
        save_path_mean = save_dir / f"{chronic_name}_mean_posterior.png"
        fig_mean.savefig(save_path_mean, dpi=120, bbox_inches='tight')
        print(f"\nSaved mean posterior to: {save_path_mean}")
        plt.close(fig_mean)

        # === FIGURE 2: Variance Posterior ===
        fig_var, axes_var = plt.subplots(1, 2, figsize=(20, 8))

        # Left: Histogram of variance
        axes_var[0].hist(var_edge_probs, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes_var[0].axvline(var_edge_probs.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean Variance: {var_edge_probs.mean():.4f}')
        axes_var[0].axvline(np.median(var_edge_probs), color='green', linestyle='--', linewidth=2,
                          label=f'Median Variance: {np.median(var_edge_probs):.4f}')
        axes_var[0].set_xlabel('Variance of Edge Existence Probability')
        axes_var[0].set_ylabel('Count')
        axes_var[0].set_title(f'Variance of Posterior Distribution (Across {len(self.all_posteriors)} inferences)')
        axes_var[0].legend()
        axes_var[0].grid(alpha=0.3)

        # Right: Graph visualization with variance (using variance as edge weights)
        # Create a copy of var_posterior with shape [E, K] for visualization
        var_posterior_vis = var_posterior.copy()
        var_posterior_normalized = var_posterior_vis / var_posterior_vis.max()  # Normalize for better color mapping
        visualize_graph(PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=var_posterior_normalized,
            skip_last_edge_type=True,
            visualize_edge_prob_threshold=0.0
        ), ax=axes_var[1])
        axes_var[1].set_title('Variance of Latent Graph Structure\n(Black dashed = power grid, Colored = variance)')

        fig_var.suptitle(f'Variance of RAPPO Posterior - {chronic_name} ({len(self.all_posteriors)} inferences)',
                       fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save variance figure
        save_path_var = save_dir / f"{chronic_name}_variance_posterior.png"
        fig_var.savefig(save_path_var, dpi=120, bbox_inches='tight')
        print(f"Saved variance posterior to: {save_path_var}")
        plt.close(fig_var)

        print(f"\n{'='*80}")
        print(f"Aggregate figures generation complete!")
        print(f"{'='*80}\n")

    def _visualize_entropy_vs_kl(self, per_edge_kl: npt.NDArray, posterior: npt.NDArray, rho_max: float):
        entropies = -np.sum(posterior * np.log(posterior + 1e-10), axis=-1)
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(entropies, per_edge_kl, alpha=0.6, color='purple')
        plt.xlabel('Edge Entropy')
        plt.ylabel('Per-Edge KL Divergence')
        plt.title(f'Edge Entropy vs. KL Divergence (RL Action #{self.rl_action_count}, Step {self.step_count})')
        plt.grid(alpha=0.3)
        save_dir = self.save_dir / "entropy_vs_kl"
        save_dir.mkdir(parents=True, exist_ok=True)
        chronic_name = self.chronic_id if self.chronic_id else "unknown"
        save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved entropy vs kl to: {save_path}")

    def _visualize_path_length(self, posterior: np.ndarray, is_aggregate: bool = False):
        gym_obs = self.gym_wrapper.cur_gym_obs

        # Powerline graph edges (the graph you want distances ON)
        powerline_edges = gym_obs[EDGE_INDEX][:, gym_obs[EDGE_MASK]]
        N = gym_obs[NODES].shape[-2]

        # Build graph with a library (NetworkX)
        # Use Graph() for undirected shortest paths; use DiGraph() if directed.
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(powerline_edges.T.astype(int))

        all_edges = fully_connected_edge_index(N)

        edges_sampled_from_posterior = []
        num_samples = 100

        # draw num_samples graphs from posterior
        for _ in range(num_samples):
            edge_exists = np.random.random(posterior.shape[0]) < posterior[:, 0]
            edges_sampled = all_edges[:, edge_exists]
            edges_sampled_from_posterior.append(edges_sampled)

        # Compute shortest paths using NetworkX
        path_lengths = []
        unreachable = 0
        random_src_nodes = np.random.choice(N, size=1000)
        random_dst_nodes = np.random.choice(N, size=1000)
        random_edges = np.stack([random_src_nodes, random_dst_nodes], axis=0)
        path_lengths_random = []

        for edges in edges_sampled_from_posterior:
            for src, dst in edges.T:
                src = int(src)
                dst = int(dst)
                try:
                    d = nx.shortest_path_length(G, source=src, target=dst)  # unweighted BFS under the hood
                    path_lengths.append(d)
                except nx.NetworkXNoPath:
                    unreachable += 1
                except nx.NodeNotFound:
                    # Shouldn't happen if nodes were added correctly, but keep it safe
                    unreachable += 1

        for src, dst in random_edges.T:
            src = int(src)
            dst = int(dst)
            try:
                d = nx.shortest_path_length(G, source=src, target=dst)  # unweighted BFS under the hood
                path_lengths_random.append(d)
            except nx.NetworkXNoPath:
                pass
            except nx.NodeNotFound:
                # Shouldn't happen if nodes were added correctly, but keep it safe
                pass

        # Plot histogram
        if not path_lengths:
            print(f"No reachable paths found. Unreachable pairs: {unreachable}")
            return

        max_d = max(
            max(path_lengths),
            max(path_lengths_random) if path_lengths_random else 0,
        )
        bins = np.arange(0, max_d + 2) - 0.5  # integer-centered bins

        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True, sharex=True)

        # Left: Histogram
        axes[0].hist(path_lengths,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Posterior-sampled path lengths")
        axes[0].axvline(np.mean(path_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean path length: {np.mean(path_lengths):.3f}')
        axes[0].axvline(np.median(path_lengths), color='green', linestyle='--', linewidth=2, label=f'Median path length: {np.median(path_lengths):.3f}')
        axes[0].set_xlabel('Path length')
        axes[0].set_ylabel('Probability')
        if is_aggregate:
            axes[0].set_title(f'Histogram of path lengths bridged by latent edges\n(sampled from mean posterior)')
        else:
            axes[0].set_title(f'Histogram of path lengths bridged by latent edges')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].xaxis.set_major_locator(MultipleLocator(1))

        # Right: Histogram
        axes[1].hist(path_lengths_random, bins=bins,
            density=True,
            alpha=0.6,
            label="Random path lengths")
        axes[1].axvline(np.mean(path_lengths_random), color='red', linestyle='--', linewidth=2, label=f'Mean path length: {np.mean(path_lengths_random):.3f}')
        axes[1].axvline(np.median(path_lengths_random), color='green', linestyle='--', linewidth=2, label=f'Median path length: {np.median(path_lengths_random):.3f}')
        axes[1].set_xlabel('Path length')
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f'Histogram of path lengths bridged by random edges')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].xaxis.set_major_locator(MultipleLocator(1))

        chronic_name = self.chronic_id if self.chronic_id else "unknown"
        if is_aggregate:
            fig.suptitle(
                f'RAPPO Shortest Paths (Mean Posterior) - {chronic_name} ({len(self.all_posteriors)} inferences)',
                fontsize=14, fontweight='bold')
        else:
            fig.suptitle(
                f'RAPPO Shortest Paths',
                fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()

        save_dir = self.save_dir / "shortest_path_dists"
        save_dir.mkdir(parents=True, exist_ok=True)
        if is_aggregate:
            save_path = save_dir / f"{chronic_name}_mean_shortest_path.png"
        else:
            save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\nSaved shortest path distribution to: {save_path}")

def run_rappo_visualization(
    checkpoint_path: str,
    policy_name: str = "reinforcement_learning_policy",
    checkpoint_name: str = "checkpoint_000010",
    env_name_override: str = None,
    num_episodes: int = 1,
    save_dir: Path = Path("results/visualizations")
):
    """Run RAPPO checkpoint with posterior visualization."""

    # Register models (required before loading checkpoint)
    ModelCatalog.register_custom_model("gnn_model", RLlibGNNModel)
    ModelCatalog.register_custom_model("ragnn_model", RLlibRAGNNModel)
    ModelCatalog.register_custom_model("nrignn_model", RLlibNRIGNNModel)

    # Load config using existing function
    config = load_config(checkpoint_path)
    env_config = config["env_config"]
    if env_name_override:
        env_config["env_name"] = env_name_override

    env_name = env_config["env_name"]

    logger.info(f"Loading RAPPO from: {checkpoint_path}")
    logger.info(f"Checkpoint: {checkpoint_name}")
    logger.info(f"Environment: {env_name}")

    # Use existing load_rllib_agent function
    rllib_agent, g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=checkpoint_path,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
        env_name=env_name,
        env_config=env_config
    )

    # Wrap with visualization
    agent = LatentGraphAnalysisAgent(
        rllib_agent=rllib_agent,
        gym_wrapper=gym_wrapper,
        analyser=PosteriorAnalyser(
            save_dir=save_dir,
            node_styles=get_node_styles(g2op_env, BusConnectivityGraphObsSpace),
            prior_for_graph_edges=config.get("relation_awareness", {}).get("prior_prob_for_graph_edge", 0.9),
            temperature=config.get("relation_awareness", {}).get("temperature", 0.5),
            num_edge_types=config.get("model", {}).get("custom_model_config", {}).get("encoder", {}).get("num_edge_types", 2)
        )
    )

    logger.info("Agent loaded! Starting episodes...\n")

    # Run episodes
    for episode in range(num_episodes):
        chronic_id = g2op_env.chronics_handler.get_name()
        obs = g2op_env.reset()
        agent.on_new_episode(chronic_id)
        done = False
        total_reward = 0

        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}/{num_episodes} - Chronic: {chronic_id}")
        print(f"{'='*80}\n")

        while not done:
            action = agent.act(obs, total_reward, done)
            obs, reward, done, info = g2op_env.step(action)
            total_reward += reward


    agent.on_evaluation_end()

    try:
        gym_wrapper.env_gym.close()
    except Exception as e:
        logger.debug(f"Environment cleanup error (ignored): {e}")


def main():
    # CONFIGURATION - Edit these parameters
    checkpoint_path = "/home/adrian/Schreibtisch/1901/1901_rappo_with_anneal_different_betas/CustomPPO_0_426b7_2026-01-19_10-28-48"
    checkpoint_name = "checkpoint_000020"
    env_name_override = "l2rpn_case14_sandbox_val"

    num_episodes = 1  # Number of episodes to run
    run_rappo_visualization(
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
        env_name_override=env_name_override,
        num_episodes=num_episodes,
        save_dir=Path("results/experiments/2601_compute_metrics"),
    )


if __name__ == "__main__":
    main()

