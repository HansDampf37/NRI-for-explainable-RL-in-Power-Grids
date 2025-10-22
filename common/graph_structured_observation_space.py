"""
This script contains observation space classes that transform the observation from grid2op into gymnasium Dict-like
observations. The dict structures the data into node-data, edge-data, global data, and edge index.
The edge index is an adjacency list of shape [2, NUM_EDGES].

This script contains three classes:
GraphObservationSpace:
This observation considers the Graph G=(V,E) where:
- V = {loads, generators, substations}
- E = {powerlines, connections from loads/generators to substations}

BipartiteGraphObservationSpace:
This observation considers the bipartit Graph G=(V+E,E') where:
- V = {loads, generators, substations}
- E = {powerlines, connections from loads/generators to substations}
- E' = {(v,e) in V x E | v == e[0] || v == e[1]}

BusConnectionsGraphObsSpace
This observation considers the graph where nodes encode connections between loads/generators/lines and buses.

"""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from grid2op.Observation import BaseObservation, ObservationSpace
from gymnasium.spaces import Dict, Box
from torch_geometric.data import Data

NODES = "node_features"
EDGES = "edge_features"
EDGE_INDEX = "edge_index"
EDGE_MASK = "edge_mask"
GLOBAL = "global_features"


class GNNObservationSpace(ABC, Dict):
    """
    Superclass for any Graph observation space. Implements a dict observation space with the following spaces:
    - NODES node features shaped [num_nodes, x_dim]
    - EDGE_INDEX node adjacency with a fix amount of edges shaped [2, max_n_edge]
    - EDGE_MASK encodes the edges within EDGE_INDEX that are non-padding shaped [num_edges]
    - EDGES (optional) edge features shaped [max_n_edge, e_dim]
    - GLOBAL (optional) global features [...]
    """
    @abstractmethod
    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, np.ndarray]:
        """
        Transforms a grid2op observation into an observation of this space.
        :param g2op_obs: The grid2op observation.
        :return: a gym-like dict observation.
        """

    def close(self):
        pass  # this is just dictated by grid2op - of course without interface...

    @property
    def num_nodes(self) -> int:
        """
        @return: the number of nodes in the observation space
        """
        return self.spaces[NODES].shape[0]

    @property
    def max_num_edges(self) -> int:
        """
        @return: the maximum number of edges in the observation space
        """
        return self.spaces[EDGE_INDEX].shape[1]

    @property
    def x_dim(self) -> int:
        """
        @return: the node feature dimensionality
        """
        return self.spaces[NODES].shape[1]

    @property
    def e_dim(self) -> Optional[int]:
        """
        @return: edge feature dimensionality if this observation space contains the EDGE space else None.
        """
        return self.spaces[EDGES].shape[1] if EDGES in self.spaces.keys() else None


class GraphObservationSpace(GNNObservationSpace):
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
        self.spaces_to_keep = spaces_to_keep or [NODES, EDGES, EDGE_INDEX, EDGE_MASK]
        super().__init__(self._dict_description_from_inputs(grid2op_observation_space, self.spaces_to_keep))
        self.edge_index = self.generate_edge_index(grid2op_observation_space)
        self.edge_mask = np.ones(self.edge_index.shape[1], dtype=np.bool)

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        if GLOBAL in self.spaces_to_keep:
            result[GLOBAL] = self.global_features_from_observation(g2op_obs)
        if EDGES in self.spaces_to_keep:
            result[EDGES] = self.edge_features_from_observation(g2op_obs)
        if NODES in self.spaces_to_keep:
            result[NODES] = self.node_features_from_observation(g2op_obs)
        if EDGE_INDEX in self.spaces_to_keep:
            result[EDGE_INDEX] = self.edge_index
        if EDGE_MASK in self.spaces_to_keep:
            result[EDGE_MASK] = self.edge_mask

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
            obs.a_or,  # the current at the origin (in A)
            obs.theta_or,  # the voltage angle at the origin (in deg)
            obs.line_or_bus,  # the bus that the origin of the line is connected to (1, 2 or -1)
            obs.p_ex,  # the power at the extremity (in MW)
            obs.q_ex,  # the reactive power at the extremity (in MVar)
            obs.a_ex,  # the current at the extremity (in A)
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

    @staticmethod
    def generate_edge_index(g2op_obs_space: ObservationSpace) -> np.ndarray:
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

    def _dict_description_from_inputs(self, grid2op_observation_space: ObservationSpace, spaces_to_keep: List[str]) -> dict:
        """
        Helper function to describe the final dict space
        """
        n_node = grid2op_observation_space.n_gen + grid2op_observation_space.n_load + grid2op_observation_space.n_sub
        n_edge = grid2op_observation_space.n_line + grid2op_observation_space.n_load + grid2op_observation_space.n_gen
        result = dict()
        if GLOBAL in spaces_to_keep:
            result[GLOBAL] = Box(low=-np.inf, high=np.inf, shape=(self.NUM_GLOBAL_FEATURES,))
        if NODES in spaces_to_keep:
            result[NODES] = Box(low=-np.inf, high=np.inf, shape=(n_node, self.NUM_FEATURES_PER_NODE))
        if EDGES in spaces_to_keep:
            result[EDGES] = Box(low=-np.inf, high=np.inf, shape=(n_edge, self.NUM_FEATURES_PER_EDGE))
        if EDGE_INDEX in spaces_to_keep:
            result[EDGE_INDEX] = Box(low=0, high=1, shape=(2, n_edge), dtype=np.long)
        if EDGE_MASK in spaces_to_keep:
            result[EDGE_MASK] = Box(low=0, high=1, shape=(n_edge, ), dtype=np.long)

        return result


class BipartitGraphObservationSpace(GNNObservationSpace):
    """
    This Observation space structures observation data similar to GraphObservationSpace in a graph-like structure.
    In contrast to GraphObservationSpace this class creates a bipartit meta-graph G=(V+E, E').
    The node set V+E contains nodes and edges from our previous graph. The edge set E' connects nodes v and e if they
    are adjacent in the original graph.
    """

    NUM_FEATURES_PER_NODE = GraphObservationSpace.NUM_FEATURES_PER_NODE + GraphObservationSpace.NUM_FEATURES_PER_EDGE

    def __init__(self, grid2op_observation_space: ObservationSpace):
        self.graph_obs_space = GraphObservationSpace(grid2op_observation_space, [NODES, EDGES, EDGE_INDEX])
        self.spaces_to_keep = [NODES, EDGE_INDEX]
        n_node_bipart = self.graph_obs_space.num_nodes + self.graph_obs_space.max_num_edges
        n_edge_bipart = 2 * self.graph_obs_space.max_num_edges

        super().__init__({
            NODES: Box(low=-np.inf, high=np.inf, shape=(n_node_bipart, self.NUM_FEATURES_PER_NODE)),
            EDGE_INDEX: Box(low=0, high=1, shape=(2, n_edge_bipart), dtype=np.long),
            EDGE_MASK: Box(low=0, high=1, shape=(n_edge_bipart, ), dtype=np.long)
        })

        edge_index = self.graph_obs_space.generate_edge_index(grid2op_observation_space)
        edge_indices = np.arange(self.graph_obs_space.max_num_edges) + self.graph_obs_space.num_nodes

        bipartit_edge_index_source = np.stack([edge_index[0], edge_indices])
        bipartit_edge_index_target = np.stack([edge_indices, edge_index[1]])
        self.bipartit_edge_index = np.concatenate([bipartit_edge_index_source, bipartit_edge_index_target], axis=1)
        self.edge_mask = np.ones((self.max_num_edges, ), dtype=np.bool)

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

        return {
            NODES: bipartit_node_features,
            EDGE_INDEX: self.bipartit_edge_index,
            EDGE_MASK: self.edge_mask,
        }


class BusConnectionsGraphObsSpace(GNNObservationSpace):
    """
    This observation space outputs a graph structured as follows:
    - loads, generators and powerline-bus-connections are modelled as nodes
    - these nodes are adjacent based on their connectivity to the buses
    Node features include:
    - active/reactive power,
    - voltage, voltage angle
    - current
    """
    NUM_FEATURES_PER_NODE = 8

    def __init__(self, grid2op_observation_space: ObservationSpace):
        obs_space = grid2op_observation_space
        num_node = obs_space.n_gen + obs_space.n_load + 2 * obs_space.n_line
        num_connections = obs_space.sub_info
        num_line = obs_space.n_line
        max_n_edge = (num_connections * (num_connections - 1)).sum() + num_line

        super().__init__({
            NODES: Box(low=-np.inf, high=np.inf, shape=(num_node, self.NUM_FEATURES_PER_NODE)),
            EDGE_INDEX: Box(low=0, high=1, shape=(2, max_n_edge), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1, shape=(max_n_edge,), dtype=np.bool)
        })

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, np.ndarray]:
        """
        Transforms the grid2op observation instance into a dict observation compatible with the gymnasium API.
        """
        node_features = self.get_node_features(g2op_obs)
        edge_index = self.get_edge_index(g2op_obs)
        # pad edge index
        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self.max_num_edges), dtype=int)
        edge_index_padded[:, :num_edges] = edge_index
        # add edge_mask
        edge_mask = np.zeros(self.max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True

        return {
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask
        }

    def get_edge_index(self, g2op_obs: BaseObservation) -> np.ndarray:
        """
        Compute the [2, E_max] shaped padded edge index. As a padding 0s are added.
        :param g2op_obs: The g2op observation
        :return: The padded edge index
        """
        connected_to_sub = np.concatenate([g2op_obs.line_or_to_subid, g2op_obs.line_ex_to_subid, g2op_obs.gen_to_subid, g2op_obs.load_to_subid])
        connected_to_bus = np.concatenate([g2op_obs.line_or_bus, g2op_obs.line_ex_bus, g2op_obs.gen_bus, g2op_obs.load_bus])

        edge_index = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if connected_to_bus[i] == connected_to_bus[j] and connected_to_sub[i] == connected_to_sub[j]:
                    # nodes share substation and bus -> edge
                    edge_index.append([i, j])

        for i in range(g2op_obs.n_line):
            # node i and i + n_line are endpoints of the same powerline and should be connected
            edge_index.append([i, i + g2op_obs.n_line])

        return np.array(edge_index).transpose()

    @staticmethod
    def get_node_features(g2op_obs: BaseObservation) -> np.ndarray:
        """
        Compute [N, X_dim]-shaped node features from a grid2op observation.
        :param g2op_obs: The g2op observation
        :return: The node features
        """
        # Compute currents for generators and loads
        P_MW = np.concatenate([g2op_obs.gen_p, g2op_obs.load_p])
        Q_MVar = np.concatenate([g2op_obs.gen_q, g2op_obs.load_q])
        V_kV = np.concatenate([g2op_obs.gen_v, g2op_obs.load_v])
        theta_deg = np.concatenate([g2op_obs.gen_theta, g2op_obs.load_theta])

        S = (P_MW + 1j * Q_MVar) * 1e6
        V_mag = V_kV * 1e3
        theta_rad = np.deg2rad(theta_deg)
        V_phasor = V_mag * (np.cos(theta_rad) + 1j * np.sin(theta_rad))
        I_phasor = np.conj(S) / (np.sqrt(3) * V_phasor)
        I_mag = np.abs(I_phasor)
        # Concatenate features for all nodes: line ends + generators + loads
        load_p, load_q, prod_p, prod_q, _ = g2op_obs.get_forecast_arrays()
        active_power_forecast = np.concatenate([np.zeros((2 * g2op_obs.n_line,)), prod_p[1], -load_p[1]])
        reactive_power_forecast = np.concatenate([np.zeros((2 * g2op_obs.n_line,)), prod_q[1], -load_q[1]])
        active_power = np.concatenate([g2op_obs.p_or, g2op_obs.p_ex, g2op_obs.gen_p, -g2op_obs.load_p])
        reactive_power = np.concatenate([g2op_obs.q_or, g2op_obs.q_ex, g2op_obs.gen_q, -g2op_obs.load_q])
        voltage = np.concatenate([g2op_obs.v_or, g2op_obs.v_ex, g2op_obs.gen_v, g2op_obs.load_v])
        voltage_angle = np.concatenate([g2op_obs.theta_or, g2op_obs.theta_ex, g2op_obs.gen_theta, g2op_obs.load_theta])
        current = np.concatenate([g2op_obs.a_or, g2op_obs.a_ex, I_mag])
        rho = np.concatenate([g2op_obs.rho, g2op_obs.rho, np.zeros((g2op_obs.n_gen + g2op_obs.n_load,))])

        features = np.array([
            active_power_forecast,
            reactive_power_forecast,
            active_power,
            reactive_power,
            voltage,
            voltage_angle,
            current,
            rho
        ]).transpose()

        return features


def gym2pytorch_geometric_data(observation: dict[str, np.ndarray]) -> Data:
    """
    Transforms an UNBATCHED gym-like dict observation into a pytorch geometric data object.
    :param observation: gym-like dict
    :return: A pytorch geometric data object
    """
    node_features = observation[NODES]
    edge_features = observation[EDGES] if EDGES in observation.keys() else None
    edge_index = observation[EDGE_INDEX][:, observation[EDGE_MASK]]
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
