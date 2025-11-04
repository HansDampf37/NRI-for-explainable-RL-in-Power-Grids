from dataclasses import dataclass
from typing import Optional, List

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from grid2op.Environment import Environment
from grid2op.PlotGrid import PlotMatplot
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from common.graph_structured_observation_space import GraphObservationSpace, BusConnectivityGraphObsSpace
from nri.utils import fully_connected_edge_index


@dataclass
class NodeStyle:
    position: np.ndarray
    color: str
    shape: str
    size: int
    label: str


@dataclass
class PlottingArgs:
    num_nodes: int
    node_styles: Optional[List[NodeStyle]] = None
    powerline_edge_index: Optional[np.ndarray] = None
    latent_edge_probs: Optional[np.ndarray] = None
    latent_edge_weight: float = 5.0
    do_weight_sweep: bool = False
    skip_last_edge_type: bool = True


def visualize_graph(args: PlottingArgs) -> Figure:
    """
    Visualize the graph including latent edges predicted by the NRI module.
    :param args: args for plotting
    :return a figure
    """
    assert args.latent_edge_probs is None or args.num_nodes * (args.num_nodes - 1) == args.latent_edge_probs.shape[0]
    fig = plt.figure(figsize=(18, 10))
    ax = plt.gca()

    # create graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(args.num_nodes))


    if args.powerline_edge_index is not None:
        for src, dst in args.powerline_edge_index.T:
            G.add_edge(int(src), int(dst), color="gray", weight=1, type="Connection")

    if args.latent_edge_probs is not None:
        cmap = plt.get_cmap("Pastel1")
        edge_index_fully_connected = fully_connected_edge_index(num_nodes=args.num_nodes)
        max_edge_type = args.latent_edge_probs.shape[1] - 1
        for edge_index, _ in enumerate(args.latent_edge_probs):
            for edge_type, _ in enumerate(args.latent_edge_probs[edge_index]):
                if not args.skip_last_edge_type or edge_type != max_edge_type:
                    weight = args.latent_edge_weight * args.latent_edge_probs[edge_index, edge_type]
                    if weight >= 1:
                        src, dst = edge_index_fully_connected[:, edge_index]
                        G.add_edge(int(src), int(dst), color=cmap(1+edge_type), weight=weight, type="Dependency")

    edge_colors = [d["color"] for (_, _, d) in G.edges(data=True)]
    edge_weights = [d["weight"] for (_, _, d) in G.edges(data=True)]

    if args.node_styles is not None:
        pos = {i: ns.position for i, ns in enumerate(args.node_styles)}
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, style="solid", arrows=False)
        shapes = set(ns.shape for ns in args.node_styles)
        for shape in shapes:
            idx = [i for i, ns in enumerate(args.node_styles) if ns.shape == shape]
            colors = [args.node_styles[i].color for i in idx]
            size = [args.node_styles[i].size for i in idx]
            nx.draw_networkx_nodes(G, pos, nodelist=idx, node_color=colors, node_shape=shape, node_size=size, ax=ax)

        _create_legend(args, G)
    else:
        # fallback: grey circular layout
        pos = nx.circular_layout(range(args.num_nodes))
        nx.draw(G, pos=pos, node_color="grey", edge_color=edge_colors, width=edge_weights, style="solid",
                with_labels=False, arrows=False)

    plt.axis("off")
    return fig


def _create_legend(args: PlottingArgs, G: nx.Graph) -> None:
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
            label=f"Connection"
        )
        edge_legend.insert(0, powerline_legend_entry)

    # Combine and draw
    plt.legend(handles=node_legend + edge_legend, loc="best", frameon=False)


def latent_edge_hist(accumulated_edge_probabilities: np.ndarray, skip_last_edge_type: bool = True):
    """
    Visualize a histogram showcasing the probability masses for different edges for any edge type except the first.

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

        def pos(sub_id: int, src_position: np.ndarray) -> np.ndarray:
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
        colors = ["gray"] * 2 * env.n_line + ["green"] * env.n_gen + ["orange"] * env.n_load + ["purple"] * env.n_storage
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
