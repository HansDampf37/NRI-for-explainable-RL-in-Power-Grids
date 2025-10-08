from collections import Counter
from typing import Optional

import networkx as nx
from matplotlib import pyplot as plt
from torch import Tensor


def visualize_latent_graph(
        typed_edge_index: Tensor,
        ground_truth_edge_index: Optional[Tensor] = None,
        skip_first_edge_type: bool = True):
    """
    Visualize latent edges predicted by the NRI module.
    :param typed_edge_index: Edge indices in shape [B, 3, E] where dimension 1 contains src, target, type
    :param ground_truth_edge_index: Edge index for ground truth edges [2, E]
    :param skip_first_edge_type: Skip first edge type when visualizing (defaults to True)
    """
    B, _, E = typed_edge_index.shape
    num_nodes = typed_edge_index[:, 0:2, :].max() + 1

    G = nx.MultiDiGraph()
    G.add_nodes_from(range(num_nodes))

    # add ground truth edges if specified
    if ground_truth_edge_index is not None:
        for src, dst in ground_truth_edge_index.transpose(1,0):
            G.add_edge(int(src), int(dst), color="gray", weight=3, edge_type="Ground Truth", style='dotted')

    # count predicted edges
    edge_counter = Counter()
    for batch in range(B):
        for edge in range(E):
            src, dst, edge_type = typed_edge_index[batch, :, edge]
            edge_counter[(int(src), int(dst), int(edge_type))] += 1

    # add predicted edges
    cmap = plt.get_cmap("Pastel1")
    max_count = max(edge_counter.values()) if len(edge_counter) > 0 else 1
    for (src, dst, t), count in edge_counter.items():
        if skip_first_edge_type and t == 0:
            continue
        color = cmap(t)
        weight = 6.0 * (count / max_count) ** 2
        G.add_edge(src, dst, color=color, weight=weight, edge_type=t, style='solid')

    # --- Draw graph ---
    fig = plt.figure(figsize=(8, 8))
    edge_colors = [d["color"] for (_, _, d) in G.edges(data=True)]
    edge_weights = [d["weight"] for (_, _, d) in G.edges(data=True)]
    edge_styles = [d["style"] for (_, _, d) in G.edges(data=True)]
    fixed_node_positions = nx.circular_layout(range(num_nodes))
    nx.draw(
        G,
        pos=fixed_node_positions,
        with_labels=True,
        node_size=300,
        node_color="lightgray",
        edge_color=edge_colors,
        width=edge_weights,
        style=edge_styles,
        arrows=True,
        connectionstyle="arc3,rad=0.2"
    )
    return fig