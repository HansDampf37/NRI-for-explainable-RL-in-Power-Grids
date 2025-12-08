"""
This script implements a baseline agent. The agent is a greedy agent meaning it will simulate several actions
returned by its _get_tested_action method and execute the one with the highest simulated reward.
Besides the heuristic actions candidates like reconnecting powerlines and doing nothing the agent receives action candidates
from topology policies. These policies may implement a RL-component to predict topological actions. The RL-component evaluates
the actions the topology policy returns the k best actions to the agent to simulate.
"""
import json
import logging
import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Optional

from grid2op.Action import BaseAction, ActionSpace, TopologySetAction
from grid2op.Agent import RecoPowerlineAgent, BaseAgent
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm


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
            # we search for topology actions if max rho is too high
            topology_actions = self.topology_policy.get_k_best_actions(observation, self.k)
            return reconnection_actions + topology_actions


def evaluate_agent(agent: BaseAgent, env: Environment, path_results: Path, num_episodes: int,
                   max_episode_length: Optional[int] = None):
    """
    Runs an agent on an environment for evaluation.
    :param agent: The agent
    :param env: the environment
    :param num_episodes: the number of episodes to run
    :param path_results: where to store the results
    :param max_episode_length: the maximum number of steps to take per episode
    :return:
    """
    logging.getLogger("grid2op.Environment.baseEnv.grid2op_Runner").disabled = True
    runner = Runner(**env.get_params_for_runner(), agentInstance=agent, agentClass=None)
    path_results.mkdir(exist_ok=True, parents=True)
    runner.run(
        nb_episode=num_episodes,
        max_iter=max_episode_length,
        path_save=path_results,
        add_detailed_output=True,
        pbar=True,
    )


def evaluate_sb3_alg(alg: BaseAlgorithm, env: Env, path_results: Path, num_episodes: int,
                     max_episode_length: Optional[int] = None):
    """
    This method evaluates a stable-baseline3 algorithm on a given gymnasium env.
    The evaluation results are stored in under path_results.
    This is different from evaluating agents in the sense that heuristic actions are not part of the evaluation.

    :param alg: The BaseAlgorithm to evaluate
    :param env: The env to evaluate on
    :param path_results: where to store the results
    :param num_episodes: the number of episodes to run
    :param max_episode_length: the maximum number of steps to take per episode
    """
    max_episode_length = max_episode_length if max_episode_length is not None else 9999999
    for _ in range(num_episodes):
        obs, info = env.reset()
        cumulative_reward = 0
        episode_length = 0
        for _ in range(max_episode_length):
            act, _ = alg.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(act)
            if done or truncated:
                break
            cumulative_reward += reward
            episode_length += 1
        name_chronic = os.path.basename(info['time_series_id'])
        base_path = Path(path_results, name_chronic)
        base_path.mkdir(parents=True, exist_ok=True)
        with open(base_path.joinpath("episode_meta.json"), 'w') as f:
            json.dump({
                "agent_seed": None,
                "chronics_max_timestep": -1,
                "cumulative_reward": cumulative_reward,
                "nb_timestep_played": episode_length
            }, f, indent=4)
