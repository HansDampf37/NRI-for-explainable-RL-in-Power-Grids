"""
This script implements a baseline agent. The agent is a greedy agent meaning it will simulate all actions returned by its
_get_tested_action method and execute the one with the highest simulated reward.
"""
import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, TypeVar, Generic

from grid2op.Environment import Environment
from grid2op.Runner import Runner
from grid2op.Action import BaseAction, ActionSpace, TopologySetAction
from grid2op.Agent import RecoPowerlineAgent
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import GymnasiumActionSpace, GymnasiumObservationSpace

TActionSpace = TypeVar('TActionSpace', bound=GymnasiumActionSpace)
TObservationSpace = TypeVar('TObservationSpace', bound=GymnasiumObservationSpace)
logger = logging.getLogger(__name__)


class TopologyPolicy(Generic[TActionSpace, TObservationSpace]):
    """
    The Topology policy is used to suggest k topology actions for a given observation.
    """

    def __init__(self, action_space: TActionSpace, observation_space: TObservationSpace):
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        """
        Returns the k best actions given by the policy in grid2op action format. The action-space and observation-space
        can be used to transform gym-like actions to a grid2op action and grid2op observation to gym like observations.

        :param observation: The observation in grid2op format.
        :param k: the number of best actions to return.
        :return: a list with the k best actions
        """
        pass


class BaselineAgent(RecoPowerlineAgent):
    """
    This particular greedy baseline will simulate the following actions:

    - Do nothing
    - Reconnections of disconnected powerlines
    - k Topology actions proposed the TopologyPolicy (if rho gets too high)
    """

    def __init__(
            self,
            g2op_action_space: ActionSpace,
            topology_policy: TopologyPolicy,
            k: int = 3,
            safe_max_rho: float = 0.95):
        """
        :param g2op_action_space: The action space
        :param topology_policy: A policy proposing topology related actions from the same action space
        :param k: the number of topology actions to consider
        :param safe_max_rho: the threshold for the load on powerlines. Only if this is exceeded we search for topology actions.
        """
        super().__init__(g2op_action_space)
        self.action_space = g2op_action_space
        self.safe_max_rho = safe_max_rho
        self.reco_powerline = RecoPowerlineAgent(g2op_action_space)
        self.topology_policy = topology_policy
        self.k = k

    def _get_tested_action(self, observation: BaseObservation) -> BaseAction:
        # try to perform reconnection or do nothing
        reconnection_actions = super()._get_tested_action(observation)

        if observation.rho.max() < self.safe_max_rho:
            # if we are safe we don't do anything except reconnections
            return reconnection_actions
        else:
            logger.info("Congestion detected. Searching for topology actions.")
            # we search for topology actions if max rho is too high
            topology_actions = self.topology_policy.get_k_best_actions(observation, self.k)
            return reconnection_actions + topology_actions


def evaluate(agent: BaselineAgent, env: Environment, path_results: Path, **runner_kwargs):
    """
    Runs an agent on an environment for evaluation.
    :param agent: The agent
    :param env: the environment
    :param path_results: where to store the results
    :param runner_kwargs: additional arguments to pass to the runner.run method
    :return:
    """
    runner = Runner(**env.get_params_for_runner(), agentInstance=agent, agentClass=None)
    res = runner.run(path_save=path_results, **runner_kwargs)

    # print results
    print("The results for the evaluated agent are:")
    for _, chron_id, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics with id {}\n".format(chron_id)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)