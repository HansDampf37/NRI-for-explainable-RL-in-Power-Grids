import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward


class MazeRLReward(BaseReward):
    """
    This reward class implements the reward formulated by Dorfer et al. in their paper ![Power Grid Congestion Management via Topology Optimization with AlphaZero](https://arxiv.org/pdf/2211.05612)
    Rewards is bounded in [0,1] and grow with decreasing max rho
    """
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.min_reward = 0
        self.max_reward = 1

    def __call__(self, action: BaseAction, env: BaseEnv, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool) -> float:
        if not is_done and not has_error:
            rho = env.current_obs.rho
            rho_max = max(rho)
            n_offline = env.n_line - sum(env.current_obs.line_status)
            if rho_max <= 1.0:
                u = max(rho_max - 0.5, 0)
            else:
                u = np.sum(rho[rho > 1] - 0.5)

            res = np.exp(-u - 0.5 * n_offline)
        else:
            res = self.reward_min
        return res
