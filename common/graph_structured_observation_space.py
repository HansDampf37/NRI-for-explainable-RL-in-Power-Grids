"""
This script contains observation space classes that transform the observation from grid2op into gymnasium Dict-like
observations. The dict structures the data into node-data, edge-data, global data, and edge index.
The edge index is an adjacency list of shape [2, NUM_EDGES].

This script contains two classes:
GraphObservationSpace:
This observation considers the Graph G=(V,E) where:
- V = {loads, generators, substations}
- E = {powerlines, connections from loads/generators to substations}

BipartiteGraphObservationSpace:
This observation considers the bipartit Graph G=(V+E,E') where:
- V = {loads, generators, substations}
- E = {powerlines, connections from loads/generators to substations}
- E' = {(v,e) in V x E | v == e[0] || v == e[1]}
"""

from typing import List, Optional

import numpy as np
from grid2op.Observation import BaseObservation, ObservationSpace
from gymnasium.spaces import Dict, Box

NODES = "node_features"
EDGES = "edge_features"
EDGE_INDEX = "edge_index"
GLOBAL = "global_features"


class GraphObservationSpace(Dict):
    """
    This Observation space implements the Dict action space from gymnasium. It returns a dict features for the following
    elements of the grid2op observation object:
    - global_features: a Box-space containing global features of the powergrid (time, date, ...)
    - edge_index: a Box-space containing the adjacency list as [2, E] array
    - node_features: a Box-space containing merged features for nodes (generators, loads, substations)
    - edge_features: a Box-space containing powerline features for edges (and zero feature vectors for powerlines connecting loads and generators to their substations)


    This assumes a static graph structure of the grid which is realistic for the grid2op scenario.
    """
    NUM_FEATURES_PER_GENERATOR = 9
    NUM_FEATURES_PER_NODE = 7
    NUM_FEATURES_PER_LOAD = 5
    NUM_FEATURES_PER_LINE = 13
    NUM_FEATURES_PER_EDGE = NUM_FEATURES_PER_LINE
    NUM_GLOBAL_FEATURES = 6

    def __init__(self, grid2op_observation_space: ObservationSpace, spaces_to_keep: Optional[List[str]] = None):
        """
        Constructor.
        :param grid2op_observation_space: original g2op observation space
        :param spaces_to_keep: which spaces to keep (global_features, node_features, edge_features, edge_index, generator_features, load_features, line_features)
        """
        self.spaces_to_keep = spaces_to_keep or [NODES, EDGES, EDGE_INDEX]
        self.n_gen = grid2op_observation_space.n_gen
        self.n_load = grid2op_observation_space.n_load
        self.n_line = grid2op_observation_space.n_line
        self.n_sub = grid2op_observation_space.n_sub
        self.n_node = self.n_load + self.n_sub + self.n_gen
        self.n_edge = self.n_line + self.n_load + self.n_gen
        self.edge_index = self._generate_edge_index(grid2op_observation_space)
        super().__init__(self._dict_description_from_inputs(self.spaces_to_keep))

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, np.ndarray]:
        """
        Transforms a grid2op observation into an observation of this space.
        :param g2op_obs: The grid2op observation.
        :return: a gym-like dict observation.
        """
        result: dict[str, np.ndarray] = {}
        if GLOBAL in self.spaces_to_keep:
            result[GLOBAL] = self.global_features_from_observation(g2op_obs)
        if EDGES in self.spaces_to_keep:
            result[EDGES] = self.edge_features_from_observation(g2op_obs)
        if NODES in self.spaces_to_keep:
            result[NODES] = self.node_features_from_observation(g2op_obs)
        if EDGE_INDEX in self.spaces_to_keep:
            result[EDGE_INDEX] = self.edge_index

        return result

    @staticmethod
    def line_features_from_observation(obs: BaseObservation) -> np.ndarray:
        """
        Returns a numpy representation of the features for lines in the observation. The returned numpy array
        is of shape (n, x) where n is the number of lines and x is the number of features per line.

        :param obs: The observation
        :return: the numpy representation of the lines in the observation
        """
        return np.array([
            obs.line_status,  # whether the line is connected
            obs.rho,  # the load on the line from 0 to 1
            obs.thermal_limit,  # the thermal limit of each line (in A)
            obs.p_or,  # the power at the origin (in MW)
            obs.q_or,  # the reactive power at the origin (in MVar)
            obs.a_or,  # the flow at the origin (in A)
            obs.theta_or,  # the voltage angle at the origin (in deg)
            obs.line_or_bus,  # the bus that the origin of the line is connected to (1, 2 or -1)
            obs.p_ex,  # the power at the extremity (in MW)
            obs.q_ex,  # the reactive power at the extremity (in MVar)
            obs.a_ex,  # the flow at the extremity (in A)
            obs.theta_ex,  # the voltage angle at the extremity (in deg)
            obs.line_ex_bus,  # the bus that the extremity of the line is connected to (1, 2 or -1)
        ]).transpose()

    @staticmethod
    def global_features_from_observation(obs: BaseObservation) -> np.ndarray:
        """
        Returns a numpy representation of global features in the observation. The returned numpy array is of shape
        (n, ) where n is the number of global features.
        :param obs: The observation
        :return: the numpy representation of the global features
        """
        return np.array([
            obs.year,
            obs.month,
            obs.day,
            obs.hour_of_day,
            obs.day_of_week,
            obs.minute_of_hour
        ])

    @staticmethod
    def node_features_from_observation(obs: BaseObservation) -> np.ndarray:
        """
        Returns a numpy representation of node features in the observation. The returned numpy array is of shape
        (n, x) where n is the number of nodes and x is the number of features per node.
        :param obs: g2op Observation
        :return: numpy representation of the node features
        """
        zeros_subs = np.zeros(shape=(obs.n_sub,))
        zeros_loads = np.zeros(shape=(obs.n_load,))

        return np.array([
            np.concatenate([zeros_subs, obs.gen_p, -obs.load_p]),
            np.concatenate([zeros_subs, obs.gen_q, -obs.load_q]),
            np.concatenate([zeros_subs, obs.gen_v, -obs.load_v]),
            np.concatenate([zeros_subs, obs.gen_theta, obs.load_theta]),
            np.concatenate([zeros_subs, obs.gen_bus, obs.load_bus]),
            np.concatenate([zeros_subs, obs.actual_dispatch, zeros_loads]),
            np.concatenate([zeros_subs, obs.curtailment_limit_mw, zeros_loads])
        ]).transpose()

    @staticmethod
    def edge_features_from_observation(obs: BaseObservation) -> np.ndarray:
        """
        Returns a numpy representation of edge features in the observation. The returned numpy array is of shape
        (n, x) where n is the number of edges and x is the number of features per edge.
        :param obs: g2op Observation
        :return: numpy representation of the edge features
        """
        line_features = GraphObservationSpace.line_features_from_observation(obs)
        lines_connecting_generators = np.zeros((obs.n_gen, GraphObservationSpace.NUM_FEATURES_PER_LINE))
        lines_connecting_loads = np.zeros((obs.n_load, GraphObservationSpace.NUM_FEATURES_PER_LINE))

        return np.concatenate([line_features, lines_connecting_generators, lines_connecting_loads], axis=0)

    def close(self):
        pass

    @staticmethod
    def _generate_edge_index(g2op_obs_space: ObservationSpace) -> np.ndarray:
        """
        Generate the edge index. The edge index models which nodes are connected by edges. It is of shape [2, E]
        containing source and target node indices for each edge.
        :param g2op_obs_space: the grid2op observation space
        :return: the edge index
        """
        powerline_source_idx = g2op_obs_space.line_or_to_subid
        powerline_target_idx = g2op_obs_space.line_ex_to_subid
        generator_idx = np.arange(g2op_obs_space.n_sub, g2op_obs_space.n_sub + g2op_obs_space.n_gen)
        generator_target_idx = g2op_obs_space.gen_to_subid
        load_source_idx = g2op_obs_space.load_to_subid
        load_idx = np.arange(g2op_obs_space.n_sub + g2op_obs_space.n_gen,
                             g2op_obs_space.n_sub + g2op_obs_space.n_gen + g2op_obs_space.n_load)

        return np.stack([
            np.concatenate([powerline_source_idx, generator_idx, load_source_idx]),
            np.concatenate([powerline_target_idx, generator_target_idx, load_idx])
        ])

    def _dict_description_from_inputs(self, spaces_to_keep: List[str]) -> dict:
        """
        Helper function to describe the final dict space
        """
        result = dict()
        if GLOBAL in spaces_to_keep:
            result[GLOBAL] = Box(low=-np.inf, high=np.inf, shape=(self.NUM_GLOBAL_FEATURES,))
        if NODES in spaces_to_keep:
            result[NODES] = Box(low=-np.inf, high=np.inf, shape=(self.n_node, self.NUM_FEATURES_PER_NODE))
        if EDGES in spaces_to_keep:
            result[EDGES] = Box(low=-np.inf, high=np.inf, shape=(self.n_edge, self.NUM_FEATURES_PER_EDGE))
        if EDGE_INDEX in spaces_to_keep:
            result[EDGE_INDEX] = Box(low=0, high=1, shape=(2, self.n_edge), dtype=np.long)

        return result


class BipartitGraphObservationSpace(Dict):
    """
    This Observation space structures observation data similar to GraphObservationSpace in a graph-like structure.
    In contrast to GraphObservationSpace this class creates a bipartit meta-graph G=(V+E, E').
    The node set V+E contains nodes and edges from our previous graph. The edge set E' connects nodes v and e if they
    are adjacent in the original graph.
    """

    NUM_FEATURES_PER_NODE = GraphObservationSpace.NUM_FEATURES_PER_NODE + GraphObservationSpace.NUM_FEATURES_PER_EDGE

    def __init__(self, grid2op_observation_space: ObservationSpace):
        self.graph_obs_space = GraphObservationSpace(grid2op_observation_space,[NODES, EDGES, EDGE_INDEX])
        self.n_node_bipart = self.graph_obs_space.n_node + self.graph_obs_space.n_edge
        self.n_edge_bipart = 2*self.graph_obs_space.n_edge
        self.spaces_to_keep = [NODES, EDGE_INDEX]

        super().__init__({
            NODES: Box(low=-np.inf, high=np.inf, shape=(self.n_node_bipart, self.NUM_FEATURES_PER_NODE)),
            EDGE_INDEX: Box(low=0, high=1, shape=(2, self.n_edge_bipart), dtype=np.long)
        })

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, np.ndarray]:
        d = self.graph_obs_space.to_gym(g2op_obs)
        node_features = d[NODES]  # [N, N_dim]
        edge_features = d[EDGES]  # [E, E_dim]
        number_nodes, number_node_features = node_features.shape  # N, N_dim
        number_edges, number_edge_features = edge_features.shape  # E, E_dim
        padding_1 = np.zeros(shape=(number_edges, number_node_features))  # [E, N_dim]
        padding_2 = np.zeros(shape=(number_nodes, number_edge_features))  # [N, E_dim]
        padded_node_features = np.concatenate([node_features, padding_1], axis=0)  # [N + E, N_dim]
        padded_edge_features = np.concatenate([padding_2, edge_features], axis=0)  # [N + E, E_dim]
        bipartit_node_features = np.concatenate([padded_node_features, padded_edge_features], axis=1)  # [N + E, N_dim + E_dim]

        edge_index = d[EDGE_INDEX]
        edge_indices = np.arange(number_edges) + number_nodes

        bipartit_edge_index_source = np.stack([edge_index[0], edge_indices])
        bipartit_edge_index_target = np.stack([edge_indices, edge_index[1]])
        bipartit_edge_index = np.concatenate([bipartit_edge_index_source, bipartit_edge_index_target], axis=1)

        return {
            NODES: bipartit_node_features,
            EDGE_INDEX: bipartit_edge_index
        }

    def close(self):
        pass