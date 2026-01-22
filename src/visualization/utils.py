import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from grid2op.Environment import Environment
from grid2op.PlotGrid import PlotMatplot
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.common.observation_space import GraphObservationSpace, BusConnectivityGraphObsSpace
from src.nri.utils import fully_connected_edge_index

logger = logging.getLogger(__name__)


@dataclass
class NodeStyle:
    position: npt.NDArray
    color: str
    shape: str
    size: int
    label: str


@dataclass
class PlottingArgs:
    num_nodes: int
    node_styles: Optional[List[NodeStyle]] = None
    powerline_edge_index: Optional[npt.NDArray] = None
    latent_edge_probs: Optional[npt.NDArray] = None
    latent_edge_weight: float = 5.0
    do_weight_sweep: bool = False
    skip_last_edge_type: bool = True
    visualize_edge_prob_threshold: float = 0.5


@dataclass
class AgentMetrics:
    label: str
    returns: List[float]
    survival_duration: List[int]


def visualize_agent_survival(datasets: List[AgentMetrics], save_to: Optional[Path] = None, show: bool = True):
    records = []
    for data in datasets:
        records.extend([{"Agent": data.label, "Survival Duration": d}
                        for d in data.survival_duration])
    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(15, 5))
    # --- Boxplot ---
    sns.boxplot(
        data=df,
        x="Agent",
        y="Survival Duration",
        hue="Agent",
        palette="muted",
        legend=False
    )

    plt.title("Survival Duration Boxplot per Agent")
    plt.xlabel("Agent")
    plt.ylabel("Time Steps")

    plt.tight_layout()

    if show:
        plt.show()

    if save_to is not None:
        plt.savefig(save_to)

    plt.close()


def compare_experiment_runs(experiment_path: Path):
    """
    This method assumes a folder structure like this:
    - experiment_name/
        - variant_1/
            - agent/
            - rl_algorithm/
        - variant_2/
            - agent/
            - rl_algorithm/

    Generates and stores root level comparison plots for the different runs.

    @param experiment_path: the path to the experiment folder
    @return: a list of figure comparing the variants
    """
    if not experiment_path.exists() or not experiment_path.is_dir():
        raise FileNotFoundError(f"Experiment folder {experiment_path} does not exist")

    if len(os.listdir(experiment_path)) == 0:
        raise FileNotFoundError(f"Experiment folder {experiment_path} does not contain any files")

    metrics = {}

    for variant in os.listdir(experiment_path):
        for sub_variant in ["agent", "rl_algorithm"]:
            variant_path = Path(experiment_path, variant, sub_variant)
            if not variant_path.exists() or not variant_path.is_dir():
                logger.warning(f"Folder {variant_path} does not exist")
                continue

            if len(os.listdir(variant_path)) == 0:
                logger.warning(f"Folder {variant_path} does not contain any files")
                continue

            for dataset in os.listdir(sub_variant):
                metrics[dataset][sub_variant][variant] = get_evaluation_metrics(
                    Path(experiment_path, variant, sub_variant, dataset),
                    variant
                )

    for dataset in metrics.keys():
        for sub_variant in metrics[dataset].keys():
            visualize_agent_survival(
                datasets=metrics[dataset][sub_variant],
                save_to=Path(experiment_path, f"compare_{sub_variant}_on_{dataset}.png"),
                show=False
            )


def visualize_performance_vs_prior(datasets: List[AgentMetrics]):
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Define figure with a GridSpec: widths [1, 2, 1]
    plt.figure(figsize=(20, 5))

    # --- Strip Plot ---
    records = []
    for data in datasets:
        records.extend([{"Agent": data.label, "Survival Duration": d} for d in data.survival_duration])
    df = pd.DataFrame(records)

    sns.boxplot(
        data=df,
        y="Survival Duration",
        x="Agent",
    )

    plt.title("Survival Duration (Strip Plot)")
    plt.ylabel("Time Steps")
    plt.xlabel("Prior Probability for powerline edges imposed on NRI encoder")
    plt.show()
    plt.close()


def visualize_agent_survival_return_relationship(datasets: List[AgentMetrics]):
    plt.figure(figsize=(10, 5))
    for data in datasets:
        sns.scatterplot(
            x=data.survival_duration,
            y=data.returns,
            label=data.label
        )
    plt.title("Returns vs Survival Duration")
    plt.xlabel("Survival Duration (Time Steps)")
    plt.ylabel("Return")
    plt.legend()
    plt.show()
    plt.close()


def get_evaluation_metrics(path: Path, agent_name: str) -> AgentMetrics:
    survival_duration = []
    returns = []
    for folder in path.iterdir():
        if folder.is_dir():
            with Path.joinpath(folder, "episode_meta.json").open() as f:
                episode_metadata = json.load(f)
                survival_duration.append(episode_metadata["nb_timestep_played"])
                returns.append(episode_metadata["cumulative_reward"])

    return AgentMetrics(agent_name, returns, survival_duration)


def get_training_progress(path: Path, agent_name: str) -> AgentMetrics:
    returns = []
    with path.open() as f:
        df = pd.read_csv(f)
        survival_duration = df["Value"].tolist()

    return AgentMetrics(agent_name, returns, survival_duration)


def display_training_progress(metrics: List[AgentMetrics], show: bool = True) -> Figure:
    fig = plt.figure(figsize=(10, 5))
    shortest_training_duration = min([len(metrics.survival_duration) for metrics in metrics])
    xs = np.arange(shortest_training_duration)
    cmap = plt.get_cmap("tab10")
    for i, metric in enumerate(metrics):
        survival_duration = metric.survival_duration[:shortest_training_duration]
        smooth_survival = smooth_curve_conv(survival_duration)
        plt.plot(xs, smooth_survival, label=metric.label, color=cmap(i))

    plt.legend()

    plt.title("Survival Duration vs Steps")
    plt.xlabel("Training Steps x 10^3")
    plt.ylabel("Survival Duration")

    if show:
        plt.show()

    return fig


def smooth_curve_conv(curve: List[float] | npt.NDArray, window: int = 5) -> npt.NDArray | List[float]:
    curve = np.asarray(curve, dtype=float)

    if window < 1:
        raise ValueError("window must be >= 1")

    kernel = np.ones(window) / window
    pad = window // 2

    # Repeat edge values outside the signal
    padded = np.pad(curve, pad_width=pad, mode="edge")

    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed

def smooth_curve(curve: List[float] | npt.NDArray, alpha=1.0) -> npt.NDArray | List[float]:
    smoothed = np.zeros_like(curve, dtype=float)
    smoothed[0] = curve[0]
    for i in range(1, len(curve)):
        smoothed[i] = alpha * curve[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def visualize_graph(args: PlottingArgs, ax=None) -> Figure:
    """
    Visualize the graph including latent edges predicted by the NRI module.
    :param args: args for plotting
    :param ax: optional matplotlib axis to draw on. If None, creates new figure.
    :return a figure (or None if ax is provided)
    """
    assert args.latent_edge_probs is None or args.num_nodes * (args.num_nodes - 1) == args.latent_edge_probs.shape[0]

    scale = 0.66

    # Create new figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(18 * scale, 10 * scale), dpi=100)
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    G = nx.MultiDiGraph()
    G.add_nodes_from(range(args.num_nodes))

    # base edges - convert to undirected by filtering out duplicate directed edges
    if args.powerline_edge_index is not None:
        seen_edges = set()
        for src, dst in args.powerline_edge_index.T:
            # Create unordered edge tuple (always smaller node first)
            edge_tuple = tuple(sorted([int(src), int(dst)]))
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                G.add_edge(int(src), int(dst), color="gray", weight=1, type="Connection")

    # latent edges
    if args.latent_edge_probs is not None:
        cmap = plt.get_cmap("Pastel1")
        edge_index_full = fully_connected_edge_index(num_nodes=args.num_nodes)
        max_type = args.latent_edge_probs.shape[1] - 1
        for e_idx, probs in enumerate(args.latent_edge_probs):
            src, dst = edge_index_full[:, e_idx]
            for t, p in enumerate(probs):
                if args.skip_last_edge_type and t == max_type:
                    continue
                # 1 for p = 0.5, args.latent_edge_weight for p = 1
                w = (2 * args.latent_edge_weight - 2) * p - (args.latent_edge_weight - 2) if p > args.visualize_edge_prob_threshold else 0
                if w >= 1:
                    G.add_edge(int(src), int(dst), color=cmap(1 + t), weight=w, type="Dependency")

    # get positions
    if args.node_styles is not None:
        pos = {i: ns.position for i, ns in enumerate(args.node_styles)}
    else:
        pos = nx.circular_layout(range(args.num_nodes))

    # separate edges
    conn_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "Connection"]
    dep_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "Dependency"]

    # draw latent edges FIRST (bottom layer) with transparency
    if dep_edges:
        lc = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in dep_edges],
            edge_color=[d["color"] for _, _, d in dep_edges],
            width=[d["weight"] for _, _, d in dep_edges],
            arrows=False,
            alpha=0.5,  # Make latent edges semi-transparent
            ax=ax
        )
        lc.set_zorder(1)

    # draw base edges ON TOP with dashed style for visibility
    if conn_edges:
        lc = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in conn_edges],
            edge_color=[d["color"] for _, _, d in conn_edges],
            width=[d["weight"] for _, _, d in conn_edges],
            arrows=False,
            style="--",  # Dashed style makes them distinguishable
            alpha=1.0,  # Fully opaque
            ax=ax
        )
        lc.set_zorder(3)  # Higher z-order to be on top

    # draw nodes (with highest z-order to be on top of all edges)
    if args.node_styles is not None:
        shapes = set(ns.shape for ns in args.node_styles)
        for shape in shapes:
            idx = [i for i, ns in enumerate(args.node_styles) if ns.shape == shape]
            node_collection = nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=idx,
                node_color=[args.node_styles[i].color for i in idx],
                node_shape=shape,
                node_size=[args.node_styles[i].size * scale for i in idx],
                ax=ax,
            )
            node_collection.set_zorder(10)  # Highest z-order to be on top
        # Create legend (pass ax if provided)
        _create_legend(args, G, ax)
    else:
        node_collection = nx.draw_networkx_nodes(G, pos, node_color="grey", ax=ax)
        node_collection.set_zorder(10)

    ax.axis("off")

    if return_fig:
        fig.tight_layout()
        return fig
    else:
        return None


def _create_legend(args: PlottingArgs, G: nx.Graph, ax=None) -> None:
    # --- Node legend ---
    unique_labels = {}
    for ns in args.node_styles:
        if ns.label not in unique_labels:
            unique_labels[ns.label] = (ns.color, ns.shape)

    node_legend = [
        Line2D(
            [0], [0],
            marker=shape,
            color='w',
            markerfacecolor=color,
            markersize=10,
            linestyle='None',
            label=label
        )
        for label, (color, shape) in unique_labels.items()
    ]

    # --- Edge legend (optional, for latent edge types) ---
    dependency_edge_colors = [d["color"] for (_, _, d) in G.edges(data=True) if d["type"] == "Dependency"]
    dependency_edge_colors_unique = set(dependency_edge_colors)
    edge_legend = [
        Line2D([0], [0],
               color=c,
               lw=2,
               label=f"Edge type {i}")
        for i, c in enumerate(dependency_edge_colors_unique)
    ]
    powerline_edge_color = [d["color"] for (_, _, d) in G.edges(data=True) if d["type"] == "Connection"]
    if len(powerline_edge_color) > 0:
        powerline_legend_entry = Line2D(
            [0], [0],
            color=powerline_edge_color[0],
            lw=2,
            linestyle='--',  # Match the dashed style
            label=f"Power Grid"
        )
        edge_legend.insert(0, powerline_legend_entry)

    # Combine and draw - use provided ax or current axes
    if ax is not None:
        ax.legend(handles=node_legend + edge_legend, loc="best", frameon=False)
    else:
        plt.legend(handles=node_legend + edge_legend, loc="best", frameon=False)


def latent_edge_hist(accumulated_edge_probabilities: npt.NDArray, skip_last_edge_type: bool = True):
    """
    Visualize a histogram showcasing the probabilities for different edges for any edge type except the first.

    :param accumulated_edge_probabilities: Edge probabilities of shape [E, num_edge_types] with probabilities for each edge - edge_type combination
    :param skip_last_edge_type: Whether to skip last edge type (default: True)
    :return: Reference to the Seaborn-styled matplotlib figure
    """
    if skip_last_edge_type:
        df = pd.DataFrame({'Edge probability': accumulated_edge_probabilities[:, :-1].sum(axis=-1).tolist()})
    else:
        df = pd.DataFrame({'Edge probability': accumulated_edge_probabilities.sum(axis=-1).tolist()})

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.histplot(df, x='Edge probability', bins=50, kde=True, color='skyblue', edgecolor='black', ax=ax)

    ax.set_title("Histogram of Latent Edge Probabilities", fontsize=18)
    ax.set_xlabel("Edge Probability", fontsize=14)
    ax.set_ylabel("Number of Edges", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    return fig


def get_node_styles(env: Environment, observation_space: type[GraphObservationSpace]) -> List[NodeStyle]:
    """
    For a given environment and observation space class, return a list of node style objects. Each node style object
    contains position, color and shape.
    :param env: the environment
    :param observation_space: the class of the observation space that dictates which entities are nodes
    :return: a list of node positions similar to the ones used by the grid2op plots
    """
    if observation_space == BusConnectivityGraphObsSpace:
        plot_helper = PlotMatplot(env.observation_space)

        r = 20.0
        layout = plot_helper._grid_layout

        def pos(sub_id: int, src_position: npt.NDArray) -> npt.NDArray:
            """Compute node offset position from source location toward target substation."""
            target_pos = np.array(layout[f"sub_{sub_id}"])
            vec = target_pos - src_position
            norm = np.linalg.norm(vec)
            if norm == 0:
                return target_pos
            return target_pos - (vec / norm) * r

        # assemble substation IDs
        sub_ids = np.concatenate([
            env.line_or_to_subid,
            env.line_ex_to_subid,
            env.gen_to_subid,
            env.load_to_subid,
            env.storage_to_subid
        ])

        # assemble corresponding source locations (each row shape [2])
        pointing_towards_locs = [
            [layout[f"sub_{sid}"] for sid in env.line_ex_to_subid],
            [layout[f"sub_{sid}"] for sid in env.line_or_to_subid],
            [layout[f"gen_{sid}_{gid}"] for gid, sid in enumerate(env.gen_to_subid)],
            [layout[f"load_{sid}_{lid}"] for lid, sid in enumerate(env.load_to_subid)],
            [layout[f"storage_{sid}_{stor_id}"] for stor_id, sid in enumerate(env.storage_to_subid)],
        ]
        # filter out empty lists
        pointing_towards_locs = np.vstack([sub for sub in pointing_towards_locs if len(sub) > 0])

        # compute final node positions as well as other properties
        positions = [pos(sid, np.array(src)) for sid, src in zip(sub_ids, pointing_towards_locs)]
        colors = ["gray"] * 2 * env.n_line + ["green"] * env.n_gen + ["orange"] * env.n_load + [
            "purple"] * env.n_storage
        shapes = ["o"] * 2 * env.n_line + ["p"] * env.n_gen + ["^"] * env.n_load + ["D"] * env.n_storage
        labels = (["Powerline-Bus-Connection"] * 2 * env.n_line + ["Generator-Bus-Connection"] * env.n_gen +
                  ["Load-Bus-Connection"] * env.n_load + ["Storage-Bus-Connection"] * env.n_storage)
        sizes = [30] * 2 * env.n_line + [120] * (env.n_load + env.n_storage + env.n_gen)

        node_styles = [
            NodeStyle(position=positions[i], color=colors[i], shape=shapes[i], label=labels[i], size=sizes[i])
            for i in range(len(positions))
        ]

        return node_styles
    else:
        raise NotImplementedError()


def visualize_posterior(latent_edge_posterior: npt.NDArray, latent_edge_prior: npt.NDArray,
                        skip_last: bool = True) -> Figure:
    assert latent_edge_posterior.shape == latent_edge_prior.shape
    assert latent_edge_posterior.ndim == 2
    assert latent_edge_posterior.shape[1] == 2

    if skip_last:
        latent_edge_prior = latent_edge_prior[:, :1]
        latent_edge_posterior = latent_edge_posterior[:, :1]

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    fig = plt.figure(figsize=(12, 7))
    bins = np.arange(0, 1.05, 0.05)

    data = pd.DataFrame({
        "value": np.concatenate([latent_edge_posterior.flatten(),
                                 latent_edge_prior.flatten()]),
        "group": ["posterior"] * len(latent_edge_posterior.flatten()) +
                 ["prior"] * len(latent_edge_prior.flatten())
    })

    sns.histplot(data=data, x="value", hue="group", bins=bins, multiple="layer", legend=True)

    plt.xlim((0, 1))
    plt.xlabel("Probability")
    plt.ylabel("Number of Edges")
    plt.title("Histogram of Latent Edge Probabilities (Prior & Posterior)")

    return fig

def visualize_prior(latent_edge_prior: npt.NDArray, skip_last: bool = True) -> Figure:

    if skip_last:
        latent_edge_prior = latent_edge_prior[:, :1]

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    fig = plt.figure(figsize=(12, 7))
    bins = np.arange(0, 1.05, 0.05)

    data = pd.DataFrame({"value": latent_edge_prior.flatten()})

    sns.histplot(data=data, x="value", bins=bins, multiple="layer")

    plt.xlim((0, 1))
    plt.xlabel("Probability")
    plt.ylabel("Number of Edges")
    plt.title("Histogram of Latent Edge Probabilities (Prior)")

    return fig
