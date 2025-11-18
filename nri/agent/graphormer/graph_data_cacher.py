from torch_geometric.data import Data

from nri.agent.graphormer.functional import precalculate_custom_attributes, precalculate_paths


class GraphDataRetriever:
    """
    Caches per-graph properties (node degrees and path lengths) using a
    hashable representation of the graph’s edge_index tensor. This avoids
    recomputation when the same graph structure appears multiple times.
    """

    def __init__(self):
        self.data = {}

    def get(self, graph_data: Data) -> Data:
        """
        Returns cached per-graph attributes if available; otherwise computes
        and stores them.

        :param graph_data: PyTorch Geometric Data object containing at least an `edge_index` tensor.
        :return: Tuple (node_degrees, path_lengths) for the given graph.
        """
        key = self._get_key(graph_data)

        if key not in self.data:
            precalculate_custom_attributes(graph_data)
            precalculate_paths(graph_data)
            self.data[key] = graph_data

        return self.data[key]

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
