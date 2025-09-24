import numpy as np
from grid2op.Observation import BaseObservation
from gymnasium.spaces import Dict


FEATURES_PER_GENERATOR = 9
FEATURES_PER_LOAD = 5
FEATURES_PER_LINE = 13

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
    def __init__(self):
        super().__init__() # TODO add box spaces with lower, higher, and shape
        self.graph_topo = None

    def to_gym(self, g2op_obs: BaseObservation) -> Dict:
        if self.graph_topo is None:
            self._generate_graph_topology(g2op_obs)

        generator_features: np.ndarray = self.generator_features_from_observation(g2op_obs)
        load_features: np.ndarray = self.load_features_from_observation(g2op_obs)
        line_features: np.ndarray = self.line_features_from_observation(g2op_obs)
        global_features: np.ndarray = self.global_features_from_observation(g2op_obs)
        return dict(
            generator_features=generator_features,
            load_features=load_features,
            line_features=line_features,
            global_features=global_features,
            graph_topo=self.graph_topo
        )

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
            obs.actual_dispatch, # the dispatch implemented by the environment (might differ to the above due to physical constraints)
            obs.curtailment_limit_mw,  # the production limit of the renewable generator (in MW)
            obs.gen_bus # the bus that this generator is connected to (1, 2 or -1)
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

    def close(self):
        pass

    @staticmethod
    def _generate_graph_topology(g2op_obs: BaseObservation):
        return dict(
            generator_to_subid=g2op_obs.gen_to_subid,
            load_to_subid=g2op_obs.load_to_subid,
            line_ex_to_subid=g2op_obs.line_ex_to_subid,
            line_or_to_subid=g2op_obs.line_or_to_subid,
        )