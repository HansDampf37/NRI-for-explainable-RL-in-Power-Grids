from typing import List, Optional

import numpy as np
from grid2op.Observation import BaseObservation
from gymnasium.spaces import Dict

FEATURES_PER_GENERATOR = 9
FEATURES_PER_NODE = 7
FEATURES_PER_LOAD = 5
FEATURES_PER_LINE = 13
FEATURES_PER_EDGE = FEATURES_PER_LINE
_default_spaces_to_keep = ["node_features", "edge_features", "edge_index"]


class GraphStructuredBoxObservationSpace(Dict):
    """
    This Observation space implements the Dict action space from gymnasium. It returns a dict features for the following
    elements of the grid2op observation object:
    - generator_features: a Box-space containing features for generators
    - load_features: a Box-space containing features for loads
    - line_features: a Box-space containing features for lines
    - global_features: a Box-space containing global features of the powergrid (time, date, ...)
    - adj_matrix: a Box-space containing adjacency matrix of the grid

    This assumes a static graph structure of the grid which is realistic for the grid2op scenario.
    """

    def __init__(self, spaces_to_keep: Optional[List[str]] = None):
        super().__init__()  # TODO add box spaces with lower, higher, and shape
        self.edge_index = None
        self.spaces_to_keep = spaces_to_keep or _default_spaces_to_keep

    def to_gym(self, g2op_obs: BaseObservation) -> Dict:
        if self.edge_index is None:
            self.edge_index = self._generate_edge_index(g2op_obs)

        result = dict()
        if "global_features" in self.spaces_to_keep:
            result["global_features"] = self.global_features_from_observation(g2op_obs)
        if "edge_features" in self.spaces_to_keep:
            result["edge_features"] = self.edge_features_from_observation(g2op_obs)
        if "node_features" in self.spaces_to_keep:
            result["node_features"] = self.node_features_from_observation(g2op_obs)
        if "line_features" in self.spaces_to_keep:
            result["line_features"] = self.line_features_from_observation(g2op_obs)
        if "generator_features" in self.spaces_to_keep:
            result["generator_features"] = self.generator_features_from_observation(g2op_obs)
        if "load_features" in self.spaces_to_keep:
            result["load_features"] = self.load_features_from_observation(g2op_obs)
        if "edge_index" in self.spaces_to_keep:
            result["edge_index"] = self.edge_index

        return result

    @staticmethod
    def generator_features_from_observation(obs: BaseObservation) -> np.ndarray:
        """
        Returns a numpy representation of the features for generators in the observation. The returned numpy array
        is of shape (n, x) where n is the number of generators and x is the number of features per generator.

        :param obs: The observation
        :return: the numpy representation of the generators in the observation
        """
        return np.array([
            obs.gen_pmax,  # the maximum power (in MW)
            obs.gen_p,  # the current power (in MW)
            obs.gen_q,  # the current reactive power (in MVar)
            obs.gen_v,  # the voltage at the bus (in kV)
            obs.gen_theta,  # the voltage angle at the bus (in deg)
            obs.target_dispatch,  # the dispatch asked for by the agent
            obs.actual_dispatch,
            # the dispatch implemented by the environment (might differ to the above due to physical constraints)
            obs.curtailment_limit_mw,  # the production limit of the renewable generator (in MW)
            obs.gen_bus  # the bus that this generator is connected to (1, 2 or -1)
        ]).transpose()

    @staticmethod
    def load_features_from_observation(obs: BaseObservation) -> np.ndarray:
        """
        Returns a numpy representation of the features for loads in the observation. The returned numpy array
        is of shape (n, x) where n is the number of loads and x is the number of features per load.

        :param obs: The observation
        :return: the numpy representation of the loads in the observation
        """
        return np.array([
            obs.load_p,  # the power (in MW)
            obs.load_q,  # the reactive power (in MVar)
            obs.load_v,  # the voltage at the bus (in kV)
            obs.load_theta,  # the voltage angle at the bus (in deg)
            obs.load_bus  # the bus that this load is connected to (1, 2 or -1)
        ]).transpose()

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
        return np.concatenate([[
            obs.year,
            obs.month,
            obs.day,
            obs.hour_of_day,
            obs.day_of_week,
            obs.minute_of_hour],
            obs.topo_vect
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
        return GraphStructuredBoxObservationSpace.line_features_from_observation(obs)

    def close(self):
        pass

    @staticmethod
    def _generate_edge_index(g2op_obs: BaseObservation) -> np.ndarray:
        """
        Generate the edge index. The edge index models which nodes are connected by edges. It is of shape [2, E]
        containing source and target node indices for each edge.
        :param g2op_obs: Some observation to get access to the grid
        :return: the edge index
        """
        powerline_source_idx = g2op_obs.line_or_to_subid
        powerline_target_idx = g2op_obs.line_ex_to_subid
        generator_idx = np.arange(g2op_obs.n_sub, g2op_obs.n_sub + g2op_obs.n_gen)
        generator_target_idx = g2op_obs.gen_to_subid
        load_source_idx = g2op_obs.load_to_subid
        load_idx = np.arange(g2op_obs.n_sub + g2op_obs.n_gen, g2op_obs.n_sub + g2op_obs.n_gen + g2op_obs.n_load)

        return np.stack([
            np.concatenate([powerline_source_idx, generator_idx, load_source_idx]),
            np.concatenate([powerline_target_idx, generator_target_idx, load_idx])
        ])
