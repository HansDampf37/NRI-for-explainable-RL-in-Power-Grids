"""
This script implements a baseline agent. The agent is a greedy agent meaning it will simulate all actions returned by its
_get_tested_action method and execute the one with the highest simulated reward.
"""
from abc import abstractmethod
from typing import List, TypeVar, Generic

from grid2op.Action import BaseAction, ActionSpace, TopologySetAction
from grid2op.Agent import RecoPowerlineAgent
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import GymnasiumActionSpace, GymnasiumObservationSpace

TActionSpace = TypeVar('TActionSpace', bound=GymnasiumActionSpace)
TObservationSpace = TypeVar('TObservationSpace', bound=GymnasiumObservationSpace)

class TopologyPolicy(Generic[TActionSpace, TObservationSpace]):
    """
    The Topology policy is used to suggest k topology actions for a given observation.
    """
    def __init__(self, action_space: TActionSpace, observation_space: TObservationSpace):
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        pass

class BaselineGNNAgent(RecoPowerlineAgent):
    """
    This particular greedy baseline will simulate the following actions:

    - Do nothing
    - Reconnections of disconnected powerlines
    - k Topology actions proposed the TopologyPolicy (if rho gets too high)
    """
    def __init__(self, action_space: ActionSpace, topology_policy: TopologyPolicy, k: int = 3, safe_max_rho: float = 0.95):
        """
        :param action_space: The action space
        :param topology_policy: A policy proposing topology related actions from the same action space
        :param k: the number of topology actions to consider
        :param safe_max_rho: the threshold for the load on powerlines. Only if this is exceeded we search for topology actions.
        """
        super().__init__(action_space)
        self.action_space = action_space
        self.safe_max_rho = safe_max_rho
        self.reco_powerline = RecoPowerlineAgent(action_space)
        self.topology_policy = topology_policy
        self.k = k

    def _get_tested_action(self, observation: BaseObservation) -> BaseAction:
        # try to perform reconnection or do nothing
        reconnection_actions = super()._get_tested_action(observation)

        if observation.rho.max() < self.safe_max_rho:
            # if we are safe we don't do anything except reconnections
            return reconnection_actions
        else:
            # we search for topology actions if max rho is too high
            topology_actions = self.topology_policy.get_k_best_actions(observation, self.k)
            return reconnection_actions + topology_actions
