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
from typing import Any, List, Optional

import numpy as np
from grid2op.Action import BaseAction, ActionSpace, TopologySetAction
from grid2op.Agent import RecoPowerlineAgent, BaseAgent
from grid2op.Environment import Environment
from grid2op.Episode import EpisodeData
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner
from grid2op.Runner.runner import runner_returned_type
from stable_baselines3.common.base_class import BaseAlgorithm

from src.common.env import G2OpGymEnv

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


class HeuristicsAgent(BaseAgent):
    """
    This agent executes heuristic rules based on the rule_config given.

    rule_config can contain:
    -   rho_threshold: float value. Activation threshold of the lower level agent.
    -   line_reco: Boolean value. If True: attempt to reconnect all disconnected power lines if
        this is beneficial for the max rho value.
    -   line_disc: Boolean value. If True: manually disconnect a line during sustained periods of
        overflow in order to avoid permanent damage. Reconnect the line back soon after the
        cooldown period ends.
    -   reset_topo: float value. Revert Threshold. If the max load rho < reset_topo, the agent
        will execute actions to revert to the reference topology.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        rule_config: dict,
    ):
        BaseAgent.__init__(self, action_space)
        self.activation_thresh = rule_config.get("activation_threshold", 0.95)
        self.line_reco = rule_config.get("line_reco", True)
        self.line_disc = rule_config.get("line_disc", False)
        self.reset_topo = rule_config.get("reset_topo", 0.5)
        self.simulate = rule_config.get("simulate", True)
        self.rho_max = 0

    def activate_agent(self, observation: BaseObservation):
        return self.rho_max > self.activation_thresh

    def act(self, observation: BaseObservation, reward: float, done : bool=False) -> BaseAction:
        current_action = self.action_space({})
        self.rho_max = (observation.rho.max() if observation.rho.max() > 0 else 2)
        if self.line_reco:
            current_action = self.reconnection_rule(observation, current_action)
        if self.reset_topo:
            current_action = self.revert_to_reference_topo(observation, current_action)
        if self.line_disc:
            current_action = self.disconnection_rule(observation, current_action=current_action)
        return current_action

    def reconnection_rule(self, observation: BaseObservation, current_action: BaseAction) -> BaseAction:
        """
        This methods reconnects all disconnected lines if this improves the current rho max values based on simulation.
        """
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if can_be_reco.any():
            (
                sim_obs,
                _,
                _,
                _,
            ) = observation.simulate(current_action)
            cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
            for id_ in (can_be_reco).nonzero()[0]:
                # reconnect all lines that improve the current action
                action = current_action + self.action_space({"set_line_status": [(id_, +1)]})
                (
                    sim_obs,
                    _,
                    _,
                    _,
                ) = observation.simulate(action)
                if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                    current_action = action
        return current_action

    def revert_to_reference_topo(self, observation: BaseObservation, current_action: BaseAction) -> BaseAction:
        if (self.rho_max < self.reset_topo) and (observation.current_step < observation.max_step-1):
            # Get all subs that are not in default topology
            subs_changed = np.unique(observation._topo_vect_to_sub[observation.topo_vect != 1])
            if len(subs_changed):
                (
                    sim_obs,
                    _,
                    _,
                    _,
                ) = observation.simulate(current_action)
                cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                # Simulate going back to reference topology for each substation that has changed
                action_options = []
                max_rhos = np.zeros(len(subs_changed))
                rewards = np.zeros(len(subs_changed))
                for i, sub in enumerate(subs_changed):
                    action = self.action_space(
                        {"set_bus": {
                            "substations_id":
                                [(sub, np.ones(observation.sub_info[sub], dtype=int))]
                        }
                        })
                    action_options.append(action)
                    sim_obs, rw, done, info = observation.simulate(current_action+action)
                    max_rhos[i] = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                    rewards[i] = rw
                if max_rhos[np.argmax(rewards)] < cur_max_rho:
                    # add the best revert action.
                    current_action += action_options[np.argmax(rewards)]
                    # print(current_action)
        return current_action

    def disconnection_rule(self, observation: BaseObservation, current_action: BaseAction) -> BaseAction:
        # This method manually disconnect a line during sustained periods of overflow in order to avoid permanent
        # damage. Reconnect the line back soon after the cooldown period ends.
        # This can help when parameters.NB_TIMESTEP_RECONNECTION > parameters.NB_TIMESTEP_COOLDOWN_LINE
        if np.any(observation.timestep_overflow > 1):
            (
                sim_obs,
                _,
                _,
                _,
            ) = observation.simulate(current_action)
            cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
            # Manually disconnect lines that are overflowed for more than 1 time step.
            id_ = observation.timestep_overflow.argmax()
            action = current_action + self.action_space({"set_line_status": [(id_, -1)]})
            (
                sim_obs,
                _,
                _,
                _,
            ) = observation.simulate(action)
            if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                # only disconnect when this benefits the current action.
                current_action = action
            # print(current_action)
        return current_action

    def simulate_combinations(self,
                              observation: BaseObservation,
                              topo_actions: List[BaseAction],
                              rb_action: BaseAction) -> BaseAction:
        if self.simulate:
            action_candidates = [rb_action] + [rb_action + ta for ta in topo_actions] + topo_actions
            best_action = rb_action
            best_rho = 2
            for action in action_candidates:
                sim_obs, _, _, _ = observation.simulate(action)
                sim_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                if sim_rho < best_rho:
                    best_rho = sim_rho
                    best_action = action
            action = best_action
        else:
            action = rb_action + topo_actions[0]
        return action

    
class BaselineAgent(HeuristicsAgent):
    """
    This particular greedy baseline will simulate the following actions:

    - Do nothing
    - Reconnections of disconnected powerlines
    - k Topology actions proposed the TopologyPolicy (if rho gets too high)
    """

    def __init__(
        self,
        g2op_action_space: ActionSpace,
        rule_config: dict,
        topo_policy: TopologyPolicy,
        k: int = 3,
    ):
        """
        :param g2op_action_space: The action space
        :param topo_policy: A policy proposing topology related actions from the same action space
        :param k: the number of topology actions to consider
        :param rule_config: contains heuristic rule descriptions
        """
        super().__init__(g2op_action_space, rule_config)
        self.topology_policy = topo_policy
        self.k = k

    def act(
        self, observation: BaseObservation, reward: float, done: bool = False
    ) -> BaseAction:
        """
        Returns a grid2op action based on a RLlib observation.
        """

        # First do rule based part of the agent, line reconnections, disconnections and reverrt topo if needed.
        rb_action = HeuristicsAgent.act(self, observation, reward, done)

        if HeuristicsAgent.activate_agent(self, observation):
            # Get action from trained RL-agent when in danger.
            topo_actions = self.topology_policy.get_k_best_actions(observation, self.k)

            action = HeuristicsAgent.simulate_combinations(self, observation, topo_actions, rb_action)
        else:
            action = rb_action

        return action


def evaluate_agent(agent: BaseAgent, env: Environment, path_results: Path, num_episodes: int,
                   max_episode_length: Optional[int] = None, verbose=True) -> List[runner_returned_type]:
    """
    Runs an agent on an environment for evaluation.
    :param agent: The agent
    :param env: the environment
    :param num_episodes: the number of episodes to run
    :param path_results: where to store the results
    :param max_episode_length: the maximum number of steps to take per episode
    :param verbose: print extra explanatory or diagnostic information
    :return: the evaluation results from the runner
    """
    logging.getLogger("grid2op.Environment.baseEnv.grid2op_Runner").disabled = True
    runner = Runner(**env.get_params_for_runner(), agentInstance=agent, agentClass=None)
    path_results.mkdir(exist_ok=True, parents=True)
    res = runner.run(
        nb_episode=num_episodes,
        max_iter=max_episode_length,
        path_save=path_results,
        add_detailed_output=True,
        pbar=verbose,
    )

    if verbose:
        # print results
        _print_runner_results(res)
        logger.info(f"Evaluation results are stored in: {path_results}")

    return res


def evaluate_sb3_alg(alg: BaseAlgorithm, env: G2OpGymEnv, path_results: Path, num_episodes: int,
                     max_episode_length: Optional[int] = None, verbose=True) -> List[runner_returned_type]:
    """
    This method evaluates a stable-baseline3 algorithm on a given gymnasium env.
    The evaluation results are stored in under path_results.
    This is different from evaluating agents in the sense that heuristic actions are not part of the evaluation.

    :param alg: The BaseAlgorithm to evaluate
    :param env: The env to evaluate on
    :param path_results: where to store the results
    :param num_episodes: the number of episodes to run
    :param max_episode_length: the maximum number of steps to take per episode
    :param verbose: print extra explanatory or diagnostic information
    :return: the evaluation results
    """
    max_episode_length = max_episode_length if max_episode_length is not None else 9999999
    results: List[runner_returned_type] = []

    # evaluate num_episodes
    for _ in range(num_episodes):
        obs, info = env.reset()
        rewards, actions, observations, episode_length = [], [], [obs], 0

        # until failure or max_episode_length
        for _ in range(max_episode_length):
            act, _ = alg.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(act)
            if done or truncated:
                break

            actions.append(act)
            observations.append(obs)
            rewards.append(reward)
            episode_length += 1

        name_chronic = os.path.basename(info['time_series_id'])
        cumulative_reward = sum(rewards)
        results.append(
            (name_chronic,
             name_chronic,
             cumulative_reward,
             episode_length,
             max_episode_length,
             EpisodeData(
                 rewards=rewards,
                 action_space=env._g2op_env.action_space, observation_space=env._g2op_env.observation_space,
                 actions=actions, observations=observations)
             )
        )

        # save to file
        base_path = Path(path_results, name_chronic)
        base_path.mkdir(parents=True, exist_ok=True)
        with open(base_path.joinpath("episode_meta.json"), 'w') as f:
            json.dump({
                "agent_seed": None,
                "chronics_max_timestep": -1,
                "cumulative_reward": cumulative_reward,
                "nb_timestep_played": episode_length
            }, f, indent=4)

    if verbose:
        # print results
        _print_runner_results(results)
        logger.info(f"Evaluation results are stored in: {path_results}")

    return results


def _print_runner_results(res: List[runner_returned_type]):
    logger.info("The results for the evaluated agent are:")
    for _, chron_id, cum_reward, nb_time_step, max_ts, data in res:
        logger.info(f"Chronics: '{chron_id}', Return: {cum_reward:.2f}, "
                    f"Survival Duration: {nb_time_step:.0f} / {max_ts:.0f}, "
                    f"Per-step-reward: {np.nan_to_num(data.rewards).mean():.2f} ± {np.nan_to_num(data.rewards).std():.2f}")
