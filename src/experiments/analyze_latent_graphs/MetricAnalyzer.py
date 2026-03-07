"""
Simplified script to run a saved RAPPO checkpoint and display the posterior distribution.

This version shows posterior statistics and graph visualizations.
Reuses existing infrastructure from evaluate_rllib_agent.py
"""
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from grid2op.Action import BaseAction
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from tabulate import tabulate

from src.common.observation_space import BusConnectivityGraphObsSpace
from src.experiments.analyze_latent_graphs.Metrics import (
    MetricVisualizer,
    PosteriorDistributionVisualizer, StepVisualizer, KLDivergenceVisualizer, EntropyVsKLVisualizer,
    DegreeDistributionVisualizer, ClusteringCoefficientVisualizer, InnerTreeNodeProbabilityVisualizer,
    SymmetryMetricVisualizer, EdgeNodeTypeVisualizer, BetweennessVisualizer,
    AllPairsShortestPathVisualizer
)
from src.experiments.analyze_latent_graphs.agent_analysis_framework import PosteriorAnalyzer, LatentGraphAnalysisAgent
from src.experiments.utils import AgentSpec, load_agent_from_spec
from src.nri.utils import fully_connected_edge_index, get_priors, get_prior_tensor
from src.visualization import get_node_styles
from src.visualization.utils import NodeStyle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PosteriorMetrics(PosteriorAnalyzer):
    """Class to analyse and visualize RAPPO posterior distributions."""

    def __init__(self, save_dir: Path, node_styles: List[NodeStyle], prior_for_graph_edges: float = 0.9,
                 temperature: float = 0.5, num_edge_types: int = 2, num_posterior_samples: int = 100,
                 enable_stepwise_viz: bool = True):
        self.save_dir = save_dir

        self.prior_for_graph_edges = prior_for_graph_edges
        self.temperature = temperature
        self.num_edge_types = num_edge_types
        num_nodes = len(node_styles)
        self.fully_connected_edge_index = fully_connected_edge_index(num_nodes).detach().cpu().numpy()
        self.num_nodes = num_nodes
        self.num_posterior_samples = num_posterior_samples
        self.enable_stepwise_viz = enable_stepwise_viz

        self.metrics: Dict[str, MetricVisualizer] = {
            "Node Degree": DegreeDistributionVisualizer(node_styles=node_styles),
            "Clustering Coefficient": ClusteringCoefficientVisualizer(node_styles=node_styles),
            "Inner Tree Node Probability": InnerTreeNodeProbabilityVisualizer(node_styles=node_styles),
            "Posterior Distribution": PosteriorDistributionVisualizer(node_styles=node_styles),
            "KL Divergence": KLDivergenceVisualizer(node_styles=node_styles),
            #"Entropy vs KL": EntropyVsKLVisualizer(node_styles=node_styles),
            "Path Length": AllPairsShortestPathVisualizer(node_styles=node_styles),
            "Symmetry Analysis": SymmetryMetricVisualizer(node_styles=node_styles),
            "Connected Node Types": EdgeNodeTypeVisualizer(node_styles=node_styles),
            #"Steps": StepVisualizer(node_styles=node_styles),
            "Betweenness Centrality": BetweennessVisualizer(node_styles=node_styles),
        }

        self.all_posteriors = []
        self.all_priors = []
        self.all_kls = []
        self.current_chronic_id = None
        self.step_count_this_episode = 0
        self.step_count_total = 0
        self.episode_count = 0

    def on_heuristic_step(self, powergrid_graph: npt.NDArray, observation: BaseObservation, environment: Environment):
        pass

    def on_rl_step(self, posterior: npt.NDArray, prior: npt.NDArray, powergrid_graph: npt.NDArray, observation: BaseObservation, _: Environment, action: BaseAction):
        self.step_count_this_episode += 1
        self.step_count_total += 1
        prior = self._get_prior(powergrid_graph, self.prior_for_graph_edges, self.temperature, self.num_edge_types)
        self.all_posteriors.append(posterior)
        self.all_priors.append(prior)

        # compute all metrics (always); only visualize and save per-step figures when enabled
        samples = self._sample_n_graphs(posterior, n=self.num_posterior_samples)
        node_mask = self._compute_mask_connected_nodes(posterior, threshold=0.5)
        for metric_name, metric in self.metrics.items():
            try:
                figure, _ = metric(
                    posterior=posterior,
                    prior=prior,
                    samples=samples,
                    powergrid_graph=powergrid_graph,
                    edge_index_fully_connected=self.fully_connected_edge_index,
                    node_mask=node_mask,
                    observation=observation,
                    visualize=self.enable_stepwise_viz,
                )
                if figure is not None:
                    save_dir = self.save_dir / "metrics" / metric_name.replace(" ", "_").lower()
                    image_save_path = save_dir / f"{self.current_chronic_id}_step_{self.step_count_total:04d}.svg"
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
        parent_dir = self.save_dir / "metrics_agg"
        parent_dir.mkdir(parents=True, exist_ok=True)

        aggregated_data_per_metric = {}
        for metric_name, metric in self.metrics.items():
            path_data = parent_dir / f"{metric_name.replace(' ', '_').lower()}.pkl"
            path_fig = parent_dir / f"{metric_name.replace(' ', '_').lower()}.svg"
            try:
                fig, data = metric.summarize(show_figure=True)
                metric.save_data(data, path_data)
                aggregated_data_per_metric[metric_name] = data
                logger.info(f"Saved aggregate data for {metric_name} to {path_data}")
                if fig is not None:
                    fig.savefig(path_fig)
                    plt.close(fig)
                    logger.info(f"Saved aggregate figure for {metric_name} to {path_fig}")
                else:
                    logger.warning(f"No figure returned by summarize() for {metric_name}")

            except Exception as e:
                logger.error(f"Error computing aggregate metric {metric_name}: {e}")
                traceback.print_exc()

        # Print summary table
        self.print_summary_table(aggregated_data_per_metric)

    def load_and_visualize(self):
        """
        Loads previously saved aggregated metric data from disk and generates visualizations without re-running the agent.
        """
        parent_dir = self.save_dir / "metrics_agg"
        for metric_name, metric_fn in self.metrics.items():
            path_data = parent_dir / f"{metric_name.replace(' ', '_').lower()}.pkl"
            path_fig = parent_dir / f"{metric_name.replace(' ', '_').lower()}.svg"
            if not path_data.exists():
                logger.warning(f"No saved data found for metric {metric_name} at {path_data}. Skipping.")
                continue
            try:
                data = metric_fn.load_data(path_data)
                figure = metric_fn._visualize(computation_result=data, aggregated=True, show_figure=True)
                if figure is not None:
                    figure.savefig(path_fig)
                    plt.close(figure)
                    logger.info(f"Visualized loaded data for {metric_name}, saved to {path_fig}")
            except Exception as e:
                logger.error(f"Error visualizing loaded metric {metric_name}: {e}")
                traceback.print_exc()

    def print_summary_table(self, aggregated_data_per_metric: Optional[Dict[str, object]] = None):
        """
        Prints a table with one row per metric for the three graphs and columns for mean, std, min, max, median.

        If aggregated_data_per_metric is not provided, tries to load data from disk.
        """
        if aggregated_data_per_metric is None:
            parent_dir = self.save_dir / "metrics_agg"
            aggregated_data_per_metric = {}
            for metric_name, metric_fn in self.metrics.items():
                path_data = parent_dir / f"{metric_name.replace(' ', '_').lower()}.pkl"
                if path_data.exists():
                    try:
                        aggregated_data_per_metric[metric_name] = metric_fn.load_data(path_data)
                    except Exception as e:
                        logger.warning(f"Could not load data for {metric_name}: {e}")

        ... # TODO

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

    def _get_prior(self, powergrid_graph: npt.NDArray, prior_for_graph_edges: float = 0.9, temperature: float = 0.5,
                   num_edge_types: int = 2) -> npt.NDArray:
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



def main():
    agent_spec = AgentSpec(
        name="RAPPO",
        load_path="/home/adrian/Schreibtisch/1901/1901_rappo_with_anneal_different_betas/CustomPPO_0_426b7_2026-01-19_10-28-48",
        checkpoint_name="checkpoint_000020",
    )
    env_name = "l2rpn_case14_sandbox_test"
    # Set to True to run the agent and compute metrics; False to load from disk and visualize only
    compute_data = True
    # Set to True to also save per-step figures (slow); False to only save aggregated figures
    enable_stepwise_viz = False
    num_episodes = 50
    max_total_duration_s = 60 * 60 * 1  # 1 hour
    save_dir = Path("results/graph_metrics")

    if compute_data:
        agent, env, gym_env = load_agent_from_spec(agent_spec=agent_spec, env_name=env_name)
        config = agent._rllib_agent.config
        analyzers_to_run = [
            PosteriorMetrics(
                save_dir=save_dir,
                node_styles=get_node_styles(env, BusConnectivityGraphObsSpace),
                prior_for_graph_edges=config.get("relation_awareness", {}).get("prior_prob_for_graph_edge", 0.9),
                temperature=config.get("relation_awareness", {}).get("temperature", 0.5),
                num_edge_types=config.get("model", {}).get("custom_model_config", {}).get("encoder", {}).get(
                    "num_edge_types", 2),
                enable_stepwise_viz=enable_stepwise_viz,
            )
        ]
        analysis_agent = LatentGraphAnalysisAgent(agent, gym_env, analyzers_to_run)
        logger.info("Agent loaded! Starting episodes...\n")
        analysis_agent.analyze(num_episodes=num_episodes, max_total_duration_s=max_total_duration_s)
    else:
        # Load saved aggregated data, re-generate figures and print summary table
        agent, env, gym_env = load_agent_from_spec(agent_spec=agent_spec, env_name=env_name)
        config = agent._rllib_agent.config
        analyzer = PosteriorMetrics(
            save_dir=save_dir,
            node_styles=get_node_styles(env, BusConnectivityGraphObsSpace),
            prior_for_graph_edges=config.get("relation_awareness", {}).get("prior_prob_for_graph_edge", 0.9),
            temperature=config.get("relation_awareness", {}).get("temperature", 0.5),
            num_edge_types=config.get("model", {}).get("custom_model_config", {}).get("encoder", {}).get(
                "num_edge_types", 2),
        )
        analyzer.load_and_visualize()
        analyzer.print_summary_table()


if __name__ == "__main__":
    main()
