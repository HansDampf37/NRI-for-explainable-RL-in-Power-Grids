from __future__ import annotations

from typing import Tuple, Dict, List

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree


def floyd_warshall_source_to_all(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if (cutoff is not None and cutoff <= level):
            break

    return node_paths, edge_paths


def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


def batched_shortest_path_distance(data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]
    relabeled_graphs = []
    shift = 0
    for i in range(len(graphs)):
        num_nodes = graphs[i].number_of_nodes()
        relabeled_graphs.append(nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)}))
        shift += num_nodes

    paths = [all_pairs_shortest_path(G) for G in relabeled_graphs]
    node_paths = {}
    edge_paths = {}

    for path in paths:
        for k, v in path[0].items():
            node_paths[k] = v
        for k, v in path[1].items():
            edge_paths[k] = v

    return node_paths, edge_paths

def precalculate_custom_attributes(data):
    """
    Precalculate some graph attributes for faster access, including:
    - in_degree of each node (tensor)
    - out_degree of each node (tensor)

    :param data: a PyG Data object
    :return: a PyG Data object with in_degree and out_degree attributes
    """

    # Calculate in_degree and out_degree
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    data.in_degree = degree(index=edge_index[1], num_nodes=num_nodes).long()
    data.out_degree = degree(index=edge_index[0], num_nodes=num_nodes).long()

    return data

def precalculate_paths(data: Data):
    """
    Precalculate node_paths and edge_paths for a data batch, along with path lengths as tensor
    and a 3D tensor of edge paths.

    :param data: a PyG Data object or Batch object
    :return: node_paths_length: tensor of shape [num_nodes, num_nodes] containing path lengths
    """

    if type(data) == Data:
        node_paths_dict, edge_paths_dict = shortest_path_distance(data)
    else:
        node_paths_dict, edge_paths_dict = batched_shortest_path_distance(data)
    
    # Create node path lengths tensor
    num_nodes = data.num_nodes
    node_paths_length = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    
    for src in node_paths_dict:
        for dst in node_paths_dict[src]:
            # Path length is len(path) - 1 (number of edges)
            # But for SpatialEncoding, we need len(path) (number of nodes)
            node_paths_length[src, dst] = len(node_paths_dict[src][dst])
            
    data.node_paths_length = node_paths_length
    return data

