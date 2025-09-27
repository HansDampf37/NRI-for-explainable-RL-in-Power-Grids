"""
This script implements a baseline agent. The agent is a greedy agent meaning it will simulate all actions returned by its
_get_tested_action method and execute the one with the highest simulated reward.
"""
import logging
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List

from grid2op.Action import BaseAction, ActionSpace, TopologySetAction
from grid2op.Agent import RecoPowerlineAgent
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner

logger = logging.getLogger(__name__)


class TopologyPolicy(ABC):
    """
    The Topology policy is used to suggest k topology actions for a given observation.
    """

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

    def _get_tested_action(self, observation: BaseObservation) -> List[BaseAction]:
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


def evaluate(agent: BaselineAgent, env: Environment, num_episodes: int, max_episode_length: int, path_results: Path):
    """
    Runs an agent on an environment for evaluation.
    :param agent: The agent
    :param env: the environment
    :param num_episodes: the number of episodes to run
    :param max_episode_length: the maximum number of steps to take per episode
    :param path_results: where to store the results
    :return:
    """
    runner = Runner(**env.get_params_for_runner(), agentInstance=agent, agentClass=None)
    res = runner.run(
        nb_episode=num_episodes,
        max_iter=max_episode_length,
        path_save=path_results,
        add_detailed_output=True,
        pbar=True
    )

    # print results
    print("The results for the evaluated agent are:")
    for _, chron_id, cum_reward, nb_time_step, max_ts, data in res:
        msg_tmp = f"\tFor chronics with id '{chron_id}'\n"
        msg_tmp += f"\t\t - return: {cum_reward:.2f}\n"
        msg_tmp += f"\t\t - rewards: {data.rewards.mean():.2f} ± {data.rewards.std():.2f}\n"
        msg_tmp += f"\t\t - number of time steps completed: {nb_time_step:.0f} / {max_ts:.0f}"
        print(msg_tmp)