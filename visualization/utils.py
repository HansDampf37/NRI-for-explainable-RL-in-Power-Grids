from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from grid2op.Environment import Environment
from grid2op.PlotGrid import PlotMatplot
from matplotlib import pyplot as plt
from torch import Tensor

from common.graph_structured_observation_space import GymnasiumObservationConverter, BusConnectionsGraphObsSpace
from nri.utils import fully_connected_edge_index


def visualize_graph(
        num_nodes: int,
        accumulated_edge_probs: Tensor,
        ground_truth_edge_index: Optional[Tensor] = None,
        skip_first_edge_type: bool = True,
        node_positions: Optional[np.ndarray] = None):
    """
    Visualize the graph including latent edges predicted by the NRI module.
    :param num_nodes: The number of nodes in the graph.
    :param accumulated_edge_probs: latent edge type probabilities in shape [N*(N-1), num_edge_types]
    :param ground_truth_edge_index: Edge index for ground truth edges [2, E]
    :param skip_first_edge_type: Skip first edge type when visualizing (defaults to True)
    :param node_positions: Node positions as numpy array shape [N, 2] where dimension 1 contains x and y. (Optional)
    """
    assert num_nodes * (num_nodes - 1) == accumulated_edge_probs.shape[0]
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(num_nodes))

    # add ground truth edges if specified
    if ground_truth_edge_index is not None:
        for src, dst in ground_truth_edge_index.transpose(1, 0):
            G.add_edge(int(src), int(dst), color="gray", weight=1, edge_type="Ground Truth", style='solid')

    # add predicted edges
    cmap = plt.get_cmap("Pastel1")
    edge_index_fully_connected = fully_connected_edge_index(num_nodes=num_nodes)
    for edge_index, _ in enumerate(accumulated_edge_probs):
        for edge_type, _ in enumerate(accumulated_edge_probs[edge_index]):
            if skip_first_edge_type and edge_type == 0:
                continue
            src, dst = edge_index_fully_connected[:, edge_index]
            weight = 5 * accumulated_edge_probs[edge_index, edge_type] ** 2
            if weight > 1:
                G.add_edge(int(src), int(dst), color=cmap(edge_type), weight=weight, edge_type=edge_type, style='solid')

    # Draw graph
    fig = plt.figure(figsize=(18, 10))
    edge_colors = [d["color"] for (_, _, d) in G.edges(data=True)]
    edge_weights = [d["weight"] for (_, _, d) in G.edges(data=True)]
    edge_styles = [d["style"] for (_, _, d) in G.edges(data=True)]
    fixed_node_positions = node_positions if node_positions is not None else nx.circular_layout(range(num_nodes))
    nx.draw(
        G,
        pos=fixed_node_positions,
        with_labels=False,
        node_size=80,
        node_color="darkgrey",
        edge_color=edge_colors,
        width=edge_weights,
        style=edge_styles,
        arrows=False,
    )
    return fig


def latent_edge_hist(
        accumulated_edge_probabilities: Tensor,
        skip_first_edge_type: bool = True):
    """
    Visualize a histogram showcasing the probability masses for different edges for any edge type except the first.

    :param accumulated_edge_probabilities: Edge probabilities of shape [E, num_edge_types] with probabilities for each edge - edge_type combination
    :param skip_first_edge_type: Whether to skip edges with type 0 (default: True)
    :return: Reference to the Seaborn-styled matplotlib figure
    """
    if skip_first_edge_type:
        df = pd.DataFrame({'Edge probability': accumulated_edge_probabilities[:, 1:].sum(dim=-1).tolist()})
    else:
        df = pd.DataFrame({'Edge probability': accumulated_edge_probabilities.sum(dim=-1).tolist()})

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.histplot(df, x='Edge probability', bins=50, kde=True, color='skyblue', edgecolor='black', ax=ax)

    ax.set_title("Histogram of Latent Edge Probabilities", fontsize=18)
    ax.set_xlabel("Edge Probability", fontsize=14)
    ax.set_ylabel("Number of Edges", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    return fig


def get_node_positions(env: Environment, observation_space: type[GymnasiumObservationConverter]):
    """
    For a given environment and observation space class, return a list of node positions.
    @param env: the environment
    @param observation_space: the class of the observation space that dictates which entities are nodes
    @return: a list of node positions similar to the ones used by the grid2op plots
    """
    if observation_space == BusConnectionsGraphObsSpace:
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
        ])

        # assemble corresponding source locations (each row shape [2])
        pointing_towards_locs = np.vstack([
            [layout[f"sub_{sid}"] for sid in env.line_ex_to_subid],
            [layout[f"sub_{sid}"] for sid in env.line_or_to_subid],
            [layout[f"gen_{sid}_{gid}"] for gid, sid in enumerate(env.gen_to_subid)],
            [layout[f"load_{sid}_{lid}"] for lid, sid in enumerate(env.load_to_subid)],
        ])

        # compute final node positions
        node_positions = np.array([pos(sid, np.array(src)) for sid, src in zip(sub_ids, pointing_towards_locs)])
        return node_positions

    else:
        raise NotImplementedError()
