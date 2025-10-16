from collections import Counter
from typing import Optional

import networkx as nx
import numpy as np
from grid2op.Environment import Environment
from grid2op.PlotGrid import PlotMatplot
from matplotlib import pyplot as plt
from torch import Tensor

from common.graph_structured_observation_space import GymnasiumObservationConverter, BusConnectionsGraphObsSpace


def visualize_latent_graph(
        typed_edge_index: Tensor,
        ground_truth_edge_index: Optional[Tensor] = None,
        skip_first_edge_type: bool = True,
        node_positions: Optional[np.ndarray] = None):
    """
    Visualize latent edges predicted by the NRI module.
    :param typed_edge_index: Edge indices in shape [3, E] where dimension 1 contains src, target, type
    :param ground_truth_edge_index: Edge index for ground truth edges [2, E]
    :param skip_first_edge_type: Skip first edge type when visualizing (defaults to True)
    :param node_positions: Node positions as numpy array shape [N, 2] where dimension 1 contains x and y. (Optional)
    """
    _, E = typed_edge_index.shape
    num_nodes = typed_edge_index[0:2, :].max() + 1

    G = nx.MultiDiGraph()
    G.add_nodes_from(range(num_nodes))

    # add ground truth edges if specified
    if ground_truth_edge_index is not None:
        for src, dst in ground_truth_edge_index.transpose(1,0):
            G.add_edge(int(src), int(dst), color="gray", weight=1, edge_type="Ground Truth", style='dotted')

    # count predicted edges
    edge_counter = Counter()
    for edge in range(E):
        src, dst, edge_type = typed_edge_index[:, edge]
        edge_counter[(int(src), int(dst), int(edge_type))] += 1

    # add predicted edges
    cmap = plt.get_cmap("Pastel1")
    max_count = max(edge_counter.values()) if len(edge_counter) > 0 else 1
    for (src, dst, t), count in edge_counter.items():
        if skip_first_edge_type and t == 0:
            continue
        color = cmap(t)
        weight = 3 * (count / max_count) ** 2
        G.add_edge(src, dst, color=color, weight=weight, edge_type=t, style='solid')

    # --- Draw graph ---
    fig = plt.figure(figsize=(8, 8))
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
        arrows=True,
    )
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