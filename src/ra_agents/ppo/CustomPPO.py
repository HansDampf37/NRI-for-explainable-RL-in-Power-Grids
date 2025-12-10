from stable_baselines3.ppo import PPO
import time
import sys
from stable_baselines3.common.utils import safe_mean

class G2OpPPO(PPO):
    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_mean_interactions", safe_mean([ep_info["i"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/action_entropy_mean", safe_mean([ep_info["action_entropy"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/unique_action_ratio_mean", safe_mean([ep_info["unique_action_ratio"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/unique_actions_mean", safe_mean([ep_info["unique_actions"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)