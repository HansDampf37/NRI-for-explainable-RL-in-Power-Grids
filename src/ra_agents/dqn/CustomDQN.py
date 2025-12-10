"""
Instead of Epsilon Greedy exploration we want to use softmax action selection. This script implements the sb3 DQN algorithm
with SoftmaxActionSelection for exploration.
"""
import sys
import time
from typing import Optional

import numpy as np

import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import safe_mean

from src.common.constants import logger


class G2OpDQN(DQN):
    def dump_logs(self) -> None:
        """
        Write log data.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_mean_interactions", safe_mean([ep_info["i"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())  # type: ignore[operator]

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)


class SoftmaxDQN(G2OpDQN):
    """
    Regular DQN algorithm using softmax action selection when exploring instead of epsilon greedy.
    Softmax action selection explores actions based on probability distribution over actions. These distributions
    come from applying softmax over the q-values. Softmax is weighted with a temperature parameter that is annealed over
    time.
    """
    def __init__(self, *args, tau_start=0.5, tau_end=0.000001, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Using softmax action selection")
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau = tau_start

    def _update_tau(self):
        """
        Update tau, the weight used for computing the softmax.
        """
        # progress ratio in [0,1]
        frac = min(self.num_timesteps / 0.8 * self._total_timesteps, 1.0)
        # linearly interpolate between start and end
        self.tau = self.tau_start + frac * (self.tau_end - self.tau_start)
        self.logger.record("rollout/tau", self.tau)

    def _sample_action(self, learning_starts: int, action_noise: Optional[ActionNoise] = None, n_envs: int = 1):
        # anneal temperature
        self._update_tau()

        # before learning starts act completely randomly
        if self.num_timesteps < learning_starts:
            actions = np.array([self.action_space.sample() for _ in range(n_envs)])
            return actions, actions

        # get q-values
        obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
        q_values = self.q_net(obs_tensor)
        if type(q_values) == tuple:
            q_values = q_values[0]

        # softmax sampling
        probs = F.softmax(q_values / self.tau, dim=1)
        actions = torch.multinomial(probs, 1).cpu().numpy().flatten()

        return actions, actions