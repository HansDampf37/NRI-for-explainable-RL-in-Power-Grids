"""
This script wraps a grid2op environment with a gymnasium API and applies heuristic actions automatically.
"""
import time
from typing import Optional, Dict, Tuple, SupportsFloat, Any

import grid2op
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import DiscreteActSpace, BoxGymObsSpace
from gymnasium import Env
from gymnasium.core import ObsType, ActType
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from lightsim2grid import LightSimBackend
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from .constants import SEED
from .rewards import BaseWithBonus

class CustomMonitor(Monitor):
    """
    This monitor logs the true episode length including heuristic actions.
    """
    def __init__(
            self,
            env: gym.Env,
            filename: Optional[str] = None,
            allow_early_resets: bool = True,
            reset_keywords: tuple[str, ...] = (),
            info_keywords: tuple[str, ...] = (),
            override_existing: bool = True,
    ):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords, override_existing)
        self.nb_steps = []

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        self.nb_steps.append(float(info['nb_steps']))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = sum(self.nb_steps)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        self.nb_steps = []
        return super().reset(**kwargs)


class G2OpGymEnv(Env):
    """
    Gymnasium-compatible wrapper for Grid2Op environments with heuristic actions.

    This class wraps a Grid2Op environment and exposes it through a standard Gymnasium interface.
    This wrapper implements the same logic as GymEnvWithRecoWithDN (automatically reconnect powerlines do nothing if load is low).
    Additionally, the do-nothing action is applied whenever the maximum line load is lower than safe_max_rho.
    """

    def __init__(self,
                 env_name: str = "l2rpn_case14_sandbox",
                 safe_max_rho: float = 0.95,
                 act_space_creation=lambda env: DiscreteActSpace(env.action_space, attr_to_keep=["set_bus"]),
                 obs_space_creation=lambda env: BoxGymObsSpace(grid2op_observation_space=env.observation_space, attr_to_keep=["rho", "p_or", "gen_p", "load_p"]),
                 seed: int = SEED):
        """
        Constructor.
        @param env_name: the name of the grid2op environment
        @param safe_max_rho: do nothing if max rho is below this value
        @param act_space_creation: lambda function that creates the action space
        @param obs_space_creation: lambda function that creates the observation space
        """
        super().__init__()
        # create env
        self._g2op_env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=BaseWithBonus)
        self._g2op_env.seed(seed)
        self._gym_env = GymEnvWithRecoDNWrapper(self._g2op_env, safe_max_rho=safe_max_rho, with_forecast=True)

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
        return self._gym_env.reset(options=options) # don't pass the seed since grid2op's GymEnv doesn't support seeding although its method suggest it

    def step(self, action):
        return self._gym_env.step(action)

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