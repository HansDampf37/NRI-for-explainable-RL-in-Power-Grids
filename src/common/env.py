"""
This module provides a Gymnasium-compatible wrapper around Grid2Op environments and integrates heuristic actions.
It aligns the environment-side heuristics with the BaselineAgent behavior to ensure consistency during training and evaluation.
"""
import logging
import time
from abc import ABC
from collections import Counter
from typing import Optional, Dict, Tuple, Callable, List

import grid2op
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.Reward import L2RPNReward
from grid2op.gym_compat import DiscreteActSpace, BoxGymObsSpace
from gymnasium import Env, Space
from l2rpn_baselines.utils import GymEnvWithHeuristics
from lightsim2grid import LightSimBackend
from stable_baselines3.common.monitor import Monitor

from .constants import SEED
from .heuristic_actions import reconnection_rule, revert_to_reference_topo, disconnection_rule

logger = logging.getLogger(__name__)


def _default_act_space(env: grid2op.Environment) -> DiscreteActSpace:
    """Create a discrete Gym action space keeping only topology actions (set_bus)."""
    return DiscreteActSpace(env.action_space, attr_to_keep=["set_bus"])


def _default_obs_space(env: grid2op.Environment) -> BoxGymObsSpace:
    """Create a boxed Gym observation space keeping selected attributes for RL."""
    return BoxGymObsSpace(grid2op_observation_space=env.observation_space,
                          attr_to_keep=["rho", "p_or", "gen_p", "load_p"])


class G2OpGymEnv(Monitor):
    """
    Gymnasium-compatible wrapper for Grid2Op environments with heuristic actions and episode logging.

    Responsibilities:
    - Build a Gym-compatible action/observation space on top of Grid2Op.
    - Wrap the underlying environment with a heuristic layer mirroring BaselineAgent rules.
    - Record per-episode metrics via Monitor, including custom info key 'nb_steps' counting heuristic steps.
    """

    def __init__(self,
                 env_name: str = "l2rpn_case14_sandbox",
                 act_space_creation: Callable[[grid2op.Environment], Space] = _default_act_space,
                 obs_space_creation: Callable[[grid2op.Environment], Space] = _default_obs_space,
                 seed: int = SEED,
                 rule_config: Optional[dict] = None,
                 curriculum_learning: Optional[List[dict]] = None):
        """
        Initialize the Gym wrapper.

        Parameters:
        :param env_name: Grid2Op environment name.
        :param act_space_creation: factory to build Gym action space from Grid2Op action space.
        :param obs_space_creation: factory to build Gym observation space from Grid2Op observation space.
        :param seed: RNG seed for the Grid2Op environment.
        :param rule_config: heuristic configuration (line_reco, line_disc, reset_topo, activation_threshold).
        """
        logging.getLogger("pandapower.convert_format").disabled = True
        Env.__init__(self)
        self._steps_agent_and_heuristic = 0
        self._steps_agent = 0
        self._actions_this_episode = []
        # create env
        g2op_env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=L2RPNReward)
        g2op_env.seed(seed)
        self._gym_env = HeuristicEnv(
            g2op_env,
            with_forecast=True,
            rule_config=rule_config,
            curriculum_learning=curriculum_learning
        )
        # this class acts as a monitor for self._gym_env
        Monitor.__init__(self, self._gym_env)

        # create observation space
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = obs_space_creation(g2op_env)
        self.observation_space = self._gym_env.observation_space
        self.g2op_observation_space = g2op_env.observation_space
        self.observation_space.seed(seed)

        # create action space
        self._gym_env.action_space.close()
        self._gym_env.action_space = act_space_creation(g2op_env)
        self.action_space = self._gym_env.action_space
        self.g2op_action_space = g2op_env.action_space
        self.action_space.seed(seed)

    @property
    def _g2op_env(self) -> grid2op.Environment:
        return self.env.init_env # access the monitored gym_env's grid2op_env

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # don't pass the seed since grid2op's GymEnv doesn't support seeding although its method suggest it
        obs, info = super().reset(options=options)
        self._steps_agent_and_heuristic = info["nb_steps"]
        self._steps_agent = 0
        self._actions_this_episode = []
        # reset heuristic counters at episode start
        if hasattr(self.env, "reset_heuristic_counters"):
            self.env.reset_heuristic_counters()
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        self._steps_agent_and_heuristic += info['nb_steps']
        self._actions_this_episode.append(int(action))
        self.total_steps += 1
        self._steps_agent += 1
        if terminated or truncated:
            ep_rew = sum(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6), # reward
                "l": self._steps_agent_and_heuristic, # length (agent + heuristics)
                "total_agent_steps": self._steps_agent, # length (only agent)
                "t": round(time.time() - self.t_start, 6), # time
            }

            # compute action diversity metrics
            if len(self._actions_this_episode) > 0:
                counts = np.array(list(Counter(self._actions_this_episode).values()), dtype=float)
                probs = counts / counts.sum()
                # Shannon entropy (nats)
                entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                unique_actions = int(counts.size)
                unique_ratio = float(unique_actions / len(self._actions_this_episode)) if len(self._actions_this_episode) > 0 else 0
            else:
                entropy = 0.0
                unique_ratio = 0.0
                unique_actions = 0

            ep_info["action_entropy"] = round(entropy, 6)
            ep_info["unique_action_ratio"] = round(unique_ratio, 6)
            ep_info["unique_actions"] = unique_actions

            # include heuristic counters for this episode
            if hasattr(self.env, "get_heuristic_counters"):
                h = self.env.get_heuristic_counters()
                ep_info["do_nothing"] = h["do_nothing"]
                ep_info["line_reconnections"] = h["line_reconnections"]
                ep_info["reset_topology"] = h["reset_topology"]
                ep_info["line_disconnections"] = h["line_disconnections"]
                ep_info["heuristic_steps"] = h["heuristic_steps"]

            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(self._steps_agent_and_heuristic)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info

        return observation, reward, terminated, truncated, info

    def do_nothing(self):
        """
        Applies the DoNothing action
        :return: observation, reward, done, truncated, info
        """
        obs, reward, done, info = self._g2op_env.step(self.g2op_action_space({}))
        return self.observation_space.to_gym(obs), reward, done, False, info

    def set_curriculum(self, level: int):
        self._gym_env.set_curriculum(level)

class GymEnvWithHeuristicsAndLogs(GymEnvWithHeuristics, ABC):
    """
    This class sole purpose is to include an attribute nb_steps in its info that models the number of steps taken on the
    environment including heuristic actions. Whenever the step method is called the env is stepped 1 + n heuristic actions.

    info[nb_steps] = 1 + n

    In order to do that it overwrites some methods without changing any logic except the addition of nb_steps
    """

    def step(self, gym_action):
        """
        Overwrite step to add nb_steps attribute to info
        """
        g2op_act_tmp = self.action_space.from_gym(gym_action)
        g2op_act = self.fix_action(g2op_act_tmp, self._previous_act)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        info['nb_steps'] = 1
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
        self._previous_act = g2op_obs
        gym_obs = self.observation_space.to_gym(g2op_obs)
        if hasattr(type(self), "_gymnasium") and type(self)._gymnasium:
            truncated = False
            return gym_obs, float(reward), done, truncated, info
        else:
            return gym_obs, float(reward), done, info

    def reset(self, *, seed=None, return_info=False, options=None):
        """
        Overwrite reset to add nb_steps attribute to info
        """
        if hasattr(type(self), "_gymnasium") and type(self)._gymnasium:
            return_info = True

        done = True
        info = {}  # no extra information provided !
        while done:
            super()._aux_reset(seed, return_info, options)  # reset the scenario
            g2op_obs = self.init_env.get_obs()  # retrieve the observation
            reward = self.init_env.reward_range[0]  # the reward at first step is always minimal

            # perform the "heuristics" steps
            info['nb_steps'] = 0
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, False, info)

            # convert back the observation to gym
            if not done:
                self._previous_act = g2op_obs
                gym_obs = self.observation_space.to_gym(g2op_obs)
                break

        if return_info:
            return gym_obs, info
        else:
            return gym_obs

    def apply_heuristics_actions(self, g2op_obs: BaseObservation, reward: float, done: bool, info: Dict) -> Tuple[BaseObservation, float, bool, Dict]:
        """
        Overwrite apply_heuristics_actions to add nb_steps attribute to info
        """
        need_action = True
        res_reward = reward

        tmp_reward = reward
        tmp_info = info
        while need_action:
            need_action = False
            g2op_actions = self.heuristic_actions(g2op_obs, tmp_reward, done, tmp_info)
            for g2op_act in g2op_actions:
                need_action = True
                tmp_obs, tmp_reward, tmp_done, tmp_info = self.init_env.step(g2op_act)
                g2op_obs = tmp_obs
                done = tmp_done

                if self._reward_cumul == "max":
                    res_reward = max(tmp_reward, res_reward)
                elif self._reward_cumul == "sum":
                    res_reward += tmp_reward
                elif self._reward_cumul == "last":
                    res_reward = tmp_reward

                info['nb_steps'] += 1

                if tmp_done:
                    break
            if done:
                break
        return g2op_obs, res_reward, done, info


class HeuristicEnv(GymEnvWithHeuristicsAndLogs):
    """
    Gym environment that applies heuristic actions according to the provided rule-configuration
    """
    def __init__(self, init_env: grid2op.Environment, with_forecast: bool=False, rule_config: Optional[dict] = None, curriculum_learning: Optional[List[dict]] = None):
        super().__init__(env_init=init_env, reward_cumul="sum", with_forecast=with_forecast)
        rule_config = rule_config or {}
        curriculum_learning = curriculum_learning or []
        self._default_parameters = init_env.parameters
        self._activation_threshold = rule_config.get("activation_threshold", 0.95)
        self._line_reco = rule_config.get("line_reco", True)
        self._line_disc = rule_config.get("line_disc", False)
        self._reset_topo = rule_config.get("reset_topo", 0.9)
        self._curriculum_learning = curriculum_learning
        # heuristic counters per episode
        self._hn_do_nothing = 0
        self._hn_line_reco = 0
        self._hn_reset_topo = 0
        self._hn_line_disc = 0
        self._hn_non_agent_steps = 0

    def reset_heuristic_counters(self):
        self._hn_do_nothing = 0
        self._hn_line_reco = 0
        self._hn_reset_topo = 0
        self._hn_line_disc = 0
        self._hn_non_agent_steps = 0

    def get_heuristic_counters(self) -> Dict[str, int]:
        return {
            "do_nothing": self._hn_do_nothing,
            "line_reconnections": self._hn_line_reco,
            "reset_topology": self._hn_reset_topo,
            "line_disconnections": self._hn_line_disc,
            "heuristic_steps": self._hn_non_agent_steps,
        }

    def heuristic_actions(self, observation: BaseObservation, reward: float, done: bool, info: Dict) -> List[BaseAction]:
        rho_max = (observation.rho.max() if observation.rho.max() > 0 else 2)
        if rho_max <= self._activation_threshold:
            initial_action = self.init_env.action_space({})
            current_action = self.apply_heuristic_additions_to_action(initial_action, observation)
            # count do-nothing if no heuristic additions modified the action
            if current_action == initial_action:
                self._hn_do_nothing += 1

            self._hn_non_agent_steps += 1
            return [current_action]
        else:
            return []

    def apply_heuristic_additions_to_action(self, current_action: BaseAction, observation: BaseObservation) -> BaseAction:
        # reconnection_rule
        if self._line_reco:
            updated = reconnection_rule(observation, current_action, self.init_env.action_space)
            # if object identity changed, a reconnection action was added
            if updated != current_action:
                self._hn_line_reco += 1
            current_action = updated
        # revert_to_reference_topo
        if self._reset_topo:
            updated = revert_to_reference_topo(observation, current_action, self.init_env.action_space, self._reset_topo)
            if updated != current_action:
                self._hn_reset_topo += 1
            current_action = updated
        # disconnection_rule
        if self._line_disc:
            updated = disconnection_rule(observation, current_action, self.init_env.action_space)
            if updated != current_action:
                self._hn_line_disc += 1
            current_action = updated
        return current_action

    def set_curriculum(self, level: int):
        logger.info(f"Change curriculum to level: {level}")
        new_params = self._curriculum_learning[level] if level < len(self._curriculum_learning) else None
        if new_params is not None:
            p = self.init_env.parameters
            p.init_from_dict(new_params)
            self.init_env.change_parameters(p)
            self.init_env.reset()
        else:
            # use default parameters
            self.init_env.change_parameters(self._default_parameters)
            self.init_env.reset()

        logger.info(f"Parameters used: {self.init_env.parameters.to_dict()}")

    def step(self, gym_action):
        g2op_act_tmp = self.action_space.from_gym(gym_action)
        g2op_act = self.apply_heuristic_additions_to_action(g2op_act_tmp, self.init_env.current_obs)
        g2op_act = self.fix_action(g2op_act, self._previous_act)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        info['nb_steps'] = 1
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
        self._previous_act = g2op_obs
        gym_obs = self.observation_space.to_gym(g2op_obs)
        if hasattr(type(self), "_gymnasium") and type(self)._gymnasium:
            truncated = False
            return gym_obs, float(reward), done, truncated, info
        else:
            return gym_obs, float(reward), done, info
