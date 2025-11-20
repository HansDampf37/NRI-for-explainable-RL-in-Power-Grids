from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

from nri.agent.graphormer.functional import get_in_out_degree, precalculate_paths


class GraphDataRetriever:
    """
    Caches per-graph properties (node degrees and path lengths) using a
    hashable representation of the graph’s edge_index tensor. This avoids
    recomputation when the same graph structure appears multiple times.
    """

    def __init__(self):
        self.data = {}

    def get(self, graph_data: Data) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns cached per-graph attributes if available; otherwise computes and stores them.

        :param graph_data: PyTorch Geometric Data object containing a (batched) graph
        :return: Tuple (node_in_degrees, node_out_degrees, path_lengths) for the given graph.
        """
        batch = graph_data.batch if graph_data.batch is not None else torch.zeros(graph_data.num_nodes).long()
        batched_edge_index = graph_data.edge_index
        edge_batch = batch[batched_edge_index[0]]
        batched_in_degrees = []
        batched_out_degrees = []
        batched_shortest_path_lengths = []

        starting_node_offset = 0
        for b in range(batch.max() + 1):
            edge_batch_mask = edge_batch == b
            edge_index = batched_edge_index[:, edge_batch_mask] - starting_node_offset
            num_nodes = torch.count_nonzero(batch == b)
            starting_node_offset += num_nodes
            _graph_data = Data(num_nodes=num_nodes, edge_index=edge_index)

            key = self._get_key(_graph_data)
            if key not in self.data:
                in_deg, out_deg = get_in_out_degree(_graph_data) # 2x[N]
                shortest_path_lengths = precalculate_paths(_graph_data)
                self.data[key] = (in_deg.to(edge_index.device),
                                  out_deg.to(edge_index.device),
                                  shortest_path_lengths.to(edge_index.device))

            in_deg, out_deg, shortest_path_lengths = self.data[key]
            batched_in_degrees.append(in_deg)
            batched_out_degrees.append(out_deg)
            batched_shortest_path_lengths.append(shortest_path_lengths)

        return (torch.cat(batched_in_degrees, dim=0),
                torch.cat(batched_out_degrees, dim=0),
                torch.stack(batched_shortest_path_lengths, dim=0))

    @staticmethod
    def _get_key(graph_data: Data):
        """
        Constructs a hashable key representing the graph structure based on
        the content of `edge_index`.

        :param graph_data: PyTorch Geometric Data object.
        :return: A tuple (dtype, shape, raw_bytes) uniquely identifying the edge_index tensor.
        """
        edge_index = graph_data.edge_index
        return (
            edge_index.dtype,
            tuple(edge_index.shape),
            edge_index.cpu().numpy().tobytes(),
        )
