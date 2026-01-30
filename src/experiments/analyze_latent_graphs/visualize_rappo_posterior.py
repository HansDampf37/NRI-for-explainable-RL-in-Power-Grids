"""
Simplified script to run a saved RAPPO checkpoint and display the posterior distribution.

This version shows posterior statistics and graph visualizations.
Reuses existing infrastructure from evaluate_rllib_agent.py
"""
import logging
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
from ray.rllib.models import ModelCatalog

from evaluate_rllib_agent import load_config, load_rllib_agent
from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX, EDGE_MASK
from src.experiments.analyze_latent_graphs.Metrics import (
    MetricVisualizer,
    PosteriorDistributionVisualizer
)
from src.nri.utils import fully_connected_edge_index, get_priors, get_prior_tensor
from src.ra_agents.RAFeatureExtractor import RLlibGNNModel, RLlibRAGNNModel, RLlibNRIGNNModel
from src.rl4pnc.evaluation.evaluation_agents import RllibAgent
from src.rl4pnc.grid2op_env.observation_converter import ObservationConverter
from src.visualization import get_node_styles
from src.visualization.utils import NodeStyle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PosteriorAnalyser:
    """Class to analyse and visualize RAPPO posterior distributions."""

    def __init__(self, save_dir: Path, node_styles: List[NodeStyle], prior_for_graph_edges: float = 0.9,
                 temperature: float = 0.5, num_edge_types: int = 2, num_posterior_samples: int = 100):
        self.save_dir = save_dir

        self.prior_for_graph_edges = prior_for_graph_edges
        self.temperature = temperature
        self.num_edge_types = num_edge_types
        num_nodes = len(node_styles)
        self.fully_connected_edge_index = fully_connected_edge_index(num_nodes).detach().cpu().numpy()
        self.num_nodes = num_nodes
        self.num_posterior_samples = num_posterior_samples

        self.metrics: Dict[str, MetricVisualizer] = {
            # "Node Degree": DegreeDistributionVisualizer(node_styles=node_styles),
            # "Clustering Coefficient": ClusteringCoefficientVisualizer(node_styles=node_styles),
            # "Inner Tree Node Probability": InnerTreeNodeProbabilityVisualizer(node_styles=node_styles),
            "Posterior Distribution": PosteriorDistributionVisualizer(node_styles=node_styles),
            # "KL Divergence": KLDivergenceVisualizer(node_styles=node_styles),
            # "Entropy vs KL": EntropyVsKLVisualizer(node_styles=node_styles),
            # "Path Length": PathLengthVisualizer(node_styles=node_styles),
            # "Symmetry Analysis": SymmetryMetricVisualizer(node_styles=node_styles),
            # "Connected Node Types": EdgeNodeTypeVisualizer(node_styles=node_styles),
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

    def _get_posterior(self) -> Any:
        posterior = self.policy_model.get_posterior()  # [B, E, K]

        if posterior.dim() == 3:
            posterior = posterior[0]  # [E, K]

        posterior_np = posterior.cpu().detach().numpy()
        return posterior_np


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
    env_config = config["evaluation_config"]["env_config"]
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
            num_edge_types=config.get("model", {}).get("custom_model_config", {}).get("encoder", {}).get(
                "num_edge_types", 2)
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

        print(f"\n{'=' * 80}")
        print(f"EPISODE {episode + 1}/{num_episodes} - Chronic: {chronic_id}")
        print(f"{'=' * 80}\n")

        while not done:
            action = agent.act(obs, total_reward, done)
            obs, reward, done, info = g2op_env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1} ended with total reward: {total_reward} after {g2op_env.nb_time_step}/{g2op_env.max_episode_duration()} steps\n")

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

    num_episodes = 50  # Number of episodes to run
    run_rappo_visualization(
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
        env_name_override=env_name_override,
        num_episodes=num_episodes,
        save_dir=Path("results/experiments/2601_compute_metrics"),
    )


if __name__ == "__main__":
    main()
