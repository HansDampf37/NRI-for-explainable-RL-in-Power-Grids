"""
Simplified script to run a saved RAPPO checkpoint and display the posterior distribution.

This version shows posterior statistics and graph visualizations.
Reuses existing infrastructure from evaluate_rllib_agent.py
"""
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from matplotlib.ticker import MultipleLocator
from networkx.algorithms.bipartite.basic import density
from numpy import ndarray
from ray.rllib.models import ModelCatalog
from torch import Tensor

from evaluate_rllib_agent import load_env_config_from_params, load_rllib_agent
from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX, EDGE_MASK, NODES
from src.nri.Sampling import GumbelSoftmax
from src.nri.utils import fully_connected_edge_index, get_priors, get_prior_tensor
from src.ra_agents.RAFeatureExtractor import RLlibGNNModel, RLlibRAGNNModel, RLlibNRIGNNModel
from src.rl4pnc.evaluation.evaluation_agents import RllibAgent
from src.rl4pnc.grid2op_env.observation_converter import ObservationConverter
from src.visualization import get_node_styles
from src.visualization import visualize_graph, PlottingArgs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RAPPOPosteriorAgent(BaseAgent):
    """Wrapper around RllibAgent that displays RAPPO posterior distribution when RL model acts."""

    def __init__(self, rllib_agent: RllibAgent, gym_wrapper: ObservationConverter, save_dir: Path):
        """Initialize RAPPOPosteriorAgent."""
        BaseAgent.__init__(self, rllib_agent.action_space)

        self.rllib_agent = rllib_agent
        self.gym_wrapper = gym_wrapper
        self.save_dir = save_dir
        self.step_count = 0
        self.rl_action_count = 0  # Count of times RL model actually computed an action

        # Get the policy model for accessing posterior
        self.policy_model = rllib_agent._rllib_agent.model

        # Get environment and node styles for graph visualization
        self.g2op_env = gym_wrapper.env_gym.init_env
        self.node_styles = get_node_styles(self.g2op_env, BusConnectivityGraphObsSpace)

        # Track chronic ID for saving files
        self.chronic_id = None

        # Track all posteriors for mean/variance computation
        self.all_posteriors = []  # List to store all posterior distributions
        self.all_priors = []  # List to store all prior distributions
        self.all_edge_masks = []  # List to store all graph edge masks
        self.all_kls = []  # List to store all KL divergence values

        # For visualization - disable interactive mode to prevent empty windows
        plt.ioff()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(20, 8))

    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        """Returns action and visualizes posterior when RL model is used."""
        self.step_count += 1
        action = self.rllib_agent.act(observation, reward, done)

        # Check if RL agent will be activated (same logic as RllibAgent)
        use_rl_component = self.rllib_agent.activate_agent(observation)
        if use_rl_component:
            self.rl_action_count += 1
            print(f"\n{'='*80}")
            print(f"Step {self.step_count}: RL AGENT ACTIVATED (ρ={self.rllib_agent.rho_max:.3f} > {self.rllib_agent.activation_thresh:.3f})")
            print(f"{'='*80}")

            self.analyse(observation)

        return action

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
        self._visualize_posterior(posterior, rho_max)
        self._visualize_kl(per_edge_kl, rho_max)
        self._visualize_entropy_vs_kl(per_edge_kl, posterior, rho_max)
        self._visualize_graph_properties(posterior, rho_max)
        self._visualize_path_length(posterior, rho_max)

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
            f'RAPPO Posterior - RL Action #{self.rl_action_count}, Step {self.step_count} (Max ρ: {rho_max:.3f})',
            fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save combined figure
        save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        self.fig.savefig(save_path, dpi=120, bbox_inches='tight')

        print(f"\nSaved to: {save_path}")

    def _print_info(self, edge_probs, posterior, rho_max):
        # Print statistics
        print(f"\n{'=' * 80}")
        print(f"RL Action #{self.rl_action_count} (Total Step {self.step_count}) | Max ρ: {rho_max:.3f}")
        print(f"{'=' * 80}")
        print(f"Posterior shape: {posterior.shape} (Edges × Edge Types)")
        print(f"\nEdge Existence Probability Statistics:")
        print(f"  Mean:   {edge_probs.mean():.4f}")
        print(f"  Std:    {edge_probs.std():.4f}")
        print(f"  Min:    {edge_probs.min():.4f}")
        print(f"  Max:    {edge_probs.max():.4f}")
        print(f"  Median: {np.median(edge_probs):.4f}")

        # Show percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = np.percentile(edge_probs, p)
            print(f"  {p:2d}th: {val:.4f}")

        # Count high-probability edges
        thresholds = [0.5, 0.7, 0.9, 0.95]
        print(f"\nEdges above threshold:")
        for thresh in thresholds:
            count = (edge_probs > thresh).sum()
            pct = 100 * count / len(edge_probs)
            print(f"  > {thresh:.2f}: {count:4d} ({pct:5.2f}%)")

        # Top edges
        print(f"\nTop 10 most confident edges:")
        top_idx = np.argsort(edge_probs)[-10:][::-1]
        for i, idx in enumerate(top_idx, 1):
            print(f"  {i:2d}. Edge {idx:4d}: {edge_probs[idx]:.4f}")

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

    def _visualize_kl(self, per_edge_kl: npt.NDArray[np.float32], rho_max: float):
        """Plot histogram of per-edge KL divergence values."""
        for ax in self.axes:
            ax.clear()

        # Left: Histogram
        self.axes[0].hist(per_edge_kl, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        self.axes[0].axvline(per_edge_kl.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean KL: {per_edge_kl.mean():.3f}')
        self.axes[0].axvline(np.median(per_edge_kl), color='green', linestyle='--', linewidth=2, label=f'Median KL: {np.median(per_edge_kl):.3f}')
        self.axes[0].set_xlabel('Per-Edge KL Divergence')
        self.axes[0].set_ylabel('Count')
        self.axes[0].set_title(f'Histogram of Per-Edge KL Divergence Values')
        self.axes[0].legend()
        self.axes[0].grid(alpha=0.3)

        gym_obs = self.gym_wrapper.cur_gym_obs
        powerline_edges = gym_obs[EDGE_INDEX][:, gym_obs[EDGE_MASK]]

        # Right: Draw graph directly into subplot using visualize_graph
        self.axes[1].clear()
        edge_weights = np.stack([per_edge_kl, np.zeros(len(per_edge_kl))]).transpose() / per_edge_kl.max()
        visualize_graph(PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=edge_weights,
            skip_last_edge_type=True
        ), ax=self.axes[1])

        self.axes[1].set_title('KL per edge Graph Structure\n(Black dashed = power grid, Colored = KL value)')

        # Save
        save_dir = self.save_dir / "kl"
        save_dir.mkdir(parents=True, exist_ok=True)
        chronic_name = self.chronic_id if self.chronic_id else "unknown"

        self.fig.suptitle(
            f'RAPPO KL - RL Action #{self.rl_action_count}, Step {self.step_count} (Max ρ: {rho_max:.3f})',
            fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save combined figure
        save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        self.fig.savefig(save_path, dpi=120, bbox_inches='tight')

        print(f"\nSaved to: {save_path}")


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

        # Compute mean and variance across inferences (axis 0)
        mean_posterior = np.mean(all_posteriors_stacked, axis=0)  # [E, K]
        var_posterior = np.var(all_posteriors_stacked, axis=0)    # [E, K]

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

    def _visualize_graph_properties(self, posterior: npt.NDArray, rho_max: float):
        """Visualize graph properties such as degree distribution."""
        gym_obs = self.gym_wrapper.cur_gym_obs
        powerline_edges = gym_obs[EDGE_INDEX][:, gym_obs[EDGE_MASK]]
        N = gym_obs[NODES].shape[-2]
        all_edges = fully_connected_edge_index(N)
        edges_sampled_from_posterior = []
        num_samples = 100
        # draw num_samples graphs from posterior
        for _ in range(num_samples):
            edge_exists = np.random.random(posterior.shape[0]) < posterior[:, 0]
            edges_sampled = all_edges[:, edge_exists]
            edges_sampled_from_posterior.append(edges_sampled)

        # for every sampled graph, compute degree distribution
        degree_distributions_posterior = []
        num_nodes = 57
        for edges in edges_sampled_from_posterior:
            degrees = np.zeros(num_nodes, dtype=int)
            for src, dst in edges.T:
                degrees[src] += 1  # only count source since graph has backwards edges too
            degree_distributions_posterior.append(degrees)

        powerline_degrees = np.zeros(num_nodes, dtype=int)
        for src, dst in powerline_edges.T:
            powerline_degrees[src] += 1 # only count source since graph has backwards edges too

        # concatenate all data to determine global range
        all_degrees = np.concatenate([
            np.concatenate(degree_distributions_posterior),
            np.concatenate([powerline_degrees])
        ])

        # integer-aligned bins (recommended for degree histograms)
        min_deg = int(np.floor(all_degrees.min()))
        max_deg = int(np.ceil(all_degrees.max()))

        bins = np.arange(min_deg, max_deg + 2) - 0.5

        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True, sharex=True)

        # Left: Histogram
        axes[0].hist(np.concatenate(degree_distributions_posterior), bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        axes[0].axvline(np.mean(degree_distributions_posterior), color='red', linestyle='--', linewidth=2, label=f'Mean degree: {np.mean(degree_distributions_posterior):.3f}')
        axes[0].axvline(np.median(degree_distributions_posterior), color='green', linestyle='--', linewidth=2, label=f'Median degree: {np.median(degree_distributions_posterior):.3f}')
        axes[0].set_xlabel('Node degree')
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'Histogram of expected node degrees (estimated by sampling {num_samples} graphs from posterior)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].xaxis.set_major_locator(MultipleLocator(1))

        # Right: Histogram
        axes[1].hist(np.concatenate([powerline_degrees]), bins=bins, color='green', edgecolor='black', alpha=0.7, density=True)
        axes[1].axvline(np.mean(powerline_degrees), color='red', linestyle='--', linewidth=2, label=f'Mean degree: {np.mean(powerline_degrees):.3f}')
        axes[1].axvline(np.median(powerline_degrees), color='green', linestyle='--', linewidth=2, label=f'Median degree: {np.median(powerline_degrees):.3f}')
        axes[1].set_xlabel('Node degree')
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f'Histogram of node degrees in powergrid graph')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].xaxis.set_major_locator(MultipleLocator(1))
        fig.suptitle(
            f'RAPPO Expected degree distribution - RL Action #{self.rl_action_count}, Step {self.step_count} (Max ρ: {rho_max:.3f})',
            fontsize=14, fontweight='bold')

        save_dir = self.save_dir / "degree_distributions"
        save_dir.mkdir(parents=True, exist_ok=True)
        chronic_name = self.chronic_id if self.chronic_id else "unknown"
        save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\nSaved degree distribution to: {save_path}")

    def _visualize_path_length(self, posterior: np.ndarray, rho_max: float):
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
        fig.suptitle(
            f'RAPPO Shortest Paths - RL Action #{self.rl_action_count}, Step {self.step_count} (Max ρ: {rho_max:.3f})',
            fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()

        save_dir = self.save_dir / "shortest_path_dists"
        save_dir.mkdir(parents=True, exist_ok=True)
        chronic_name = self.chronic_id if self.chronic_id else "unknown"
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
    env_config = load_env_config_from_params(checkpoint_path)
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
    agent = RAPPOPosteriorAgent(
        rllib_agent=rllib_agent,
        gym_wrapper=gym_wrapper,
        save_dir=save_dir
    )

    logger.info("Agent loaded! Starting episodes...\n")

    # Run episodes
    for episode in range(num_episodes):
        obs = g2op_env.reset()
        done = False
        total_reward = 0
        step = 0

        # Reset counters and get chronic ID for this episode
        agent.rl_action_count = 0
        agent.step_count = 0
        agent.chronic_id = g2op_env.chronics_handler.get_name()

        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}/{num_episodes} - Chronic: {agent.chronic_id}")
        print(f"{'='*80}\n")

        while not done:
            action = agent.act(obs, total_reward, done)
            obs, reward, done, info = g2op_env.step(action)
            total_reward += reward
            step += 1

            if done:
                print(f"\n{'='*80}")
                print(f"Episode {episode + 1} finished!")
                print(f"  Total steps: {step}")
                print(f"  RL actions:  {agent.rl_action_count} ({100*agent.rl_action_count/step:.1f}%)")
                print(f"  Heuristic actions: {step - agent.rl_action_count} ({100*(step-agent.rl_action_count)/step:.1f}%)")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Reason: {info.get('exception', 'Completed')}")
                print(f"{'='*80}\n")

    # Generate aggregate figures (mean and variance) after all episodes
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.generate_aggregate_figures(save_dir)

    logger.info("Visualization complete!")
    logger.info(f"Saved to: {save_dir}")


    # Cleanup - reuse try/except pattern from existing code
    try:
        gym_wrapper.env_gym.close()
    except Exception as e:
        logger.debug(f"Environment cleanup error (ignored): {e}")


def main():
    """Main entry point."""

    # CONFIGURATION - Edit these parameters
    checkpoint_path = "/home/adrian/Schreibtisch/1901/1901_rappo_with_anneal_different_betas/CustomPPO_0_426b7_2026-01-19_10-28-48"
    checkpoint_name = "checkpoint_000020"
    env_name_override = "l2rpn_case14_sandbox_val"  # Use None to auto-load from params.json

    num_episodes = 2  # Number of episodes to run
    # NOTE: visualize_every_n means "every N times the RL MODEL acts" (not every N steps)
    # Since heuristics handle most actions, the RL model only acts when rho > threshold


    run_rappo_visualization(
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
        env_name_override=env_name_override,
        num_episodes=num_episodes,
        save_dir=Path("results/visualizations/1901_test"),
    )


if __name__ == "__main__":
    main()

