"""
This script wraps a grid2op environment with a gymnasium API and applies heuristic actions automatically.
"""
import logging
import time
from typing import Optional, Dict, Tuple, Callable
from collections import Counter

import numpy as np

import grid2op
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import DiscreteActSpace, BoxGymObsSpace
from gymnasium import Env, Space
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from lightsim2grid import LightSimBackend
from stable_baselines3.common.monitor import Monitor

from .constants import SEED
from .rewards import MazeRLReward

def _default_act_space(env: grid2op.Environment) -> DiscreteActSpace:
    return DiscreteActSpace(env.action_space, attr_to_keep=["set_bus"])

def _default_obs_space(env: grid2op.Environment) -> BoxGymObsSpace:
    return BoxGymObsSpace(grid2op_observation_space=env.observation_space, attr_to_keep=["rho", "p_or", "gen_p", "load_p"])

class G2OpGymEnv(Monitor):
    """
    Gymnasium-compatible wrapper for Grid2Op environments with heuristic actions.

    This class wraps a Grid2Op environment and exposes it through a standard Gymnasium interface.
    This wrapper implements the same logic as GymEnvWithRecoWithDN (automatically reconnect powerlines do nothing if load is low).
    Additionally, the do-nothing action is applied whenever the maximum line load is lower than safe_max_rho.
    """

    def __init__(self,
                 env_name: str = "l2rpn_case14_sandbox",
                 safe_max_rho: float = 0.95,
                 act_space_creation: Callable[[grid2op.Environment], Space] = _default_act_space,
                 obs_space_creation: Callable[[grid2op.Environment], Space] = _default_obs_space,
                 seed: int = SEED):
        """
        Constructor.
        @param env_name: the name of the grid2op environment
        @param safe_max_rho: do nothing if max rho is below this value
        @param act_space_creation: lambda function that creates the action space
        @param obs_space_creation: lambda function that creates the observation space
        """
        logging.getLogger("pandapower.convert_format").disabled = True
        Env.__init__(self)
        self._ep_len = 0
        self._interactions = 0
        self._episode_actions = []
        # create env
        self._g2op_env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=MazeRLReward)
        self._g2op_env.seed(seed)
        self._gym_env = GymEnvWithRecoDNWrapper(self._g2op_env, safe_max_rho=safe_max_rho, with_forecast=True)
        Monitor.__init__(self, self._gym_env)

        # create observation space
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = obs_space_creation(self._g2op_env)
        self.observation_space = self._gym_env.observation_space
        self.g2op_observation_space = self._g2op_env.observation_space
        self.observation_space.seed(seed)

        # create action space
        self._gym_env.action_space.close()
        self._gym_env.action_space = act_space_creation(self._g2op_env)
        self.action_space = self._gym_env.action_space
        self.g2op_action_space = self._g2op_env.action_space
        self.action_space.seed(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # don't pass the seed since grid2op's GymEnv doesn't support seeding although its method suggest it
        obs, info = super().reset(options=options)
        self._ep_len = int(info["nb_steps"])
        self._interactions = 0
        self._episode_actions = []
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        self._ep_len += int(info['nb_steps'])
        self._episode_actions.append(action)
        if terminated or truncated:
            ep_rew = sum(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": self._ep_len, "t": round(time.time() - self.t_start, 6),
                       "i": self._interactions}

            # compute action diversity metrics
            if len(self._episode_actions) > 0:
                counts = np.array(list(Counter(self._episode_actions).values()), dtype=float)
                probs = counts / counts.sum()
                # Shannon entropy (nats)
                entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                unique_actions = int(counts.size)
                unique_ratio = float(unique_actions / len(self._episode_actions)) if len(self._episode_actions) > 0 else 0
            else:
                entropy = 0.0
                unique_ratio = 0.0
                unique_actions = 0

            ep_info["action_entropy"] = round(entropy, 6)
            ep_info["unique_action_ratio"] = round(unique_ratio, 6)
            ep_info["unique_actions"] = unique_actions

            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(self._ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        self._interactions += 1
        return observation, reward, terminated, truncated, info

    def do_nothing(self):
        """
        Applies the DoNothing action
        :return: observation, reward, done, truncated, info
        """
        obs, reward, done, info = self._g2op_env.step(self._g2op_env.action_space({}))
        return self.observation_space.to_gym(obs), reward, done, False, info


class GymEnvWithRecoDNWrapper(GymEnvWithRecoWithDN):
    def step(self, gym_action):
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
