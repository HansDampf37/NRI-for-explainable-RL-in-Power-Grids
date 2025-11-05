from typing import Dict

import hydra
import torch
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace
from gymnasium.spaces import Discrete, Box
from omegaconf import DictConfig, OmegaConf
from ray.rllib.algorithms.dqn import DQNConfig

from common import G2OpGymEnv


class RayEnv(G2OpGymEnv):
    def __init__(self, env_config: Dict):
        super().__init__(
            env_name=env_config.get("env_name", "l2rpn_case14_sandbox"),
            safe_max_rho=env_config.get("safe_max_rho", 0.95),
            act_space_creation=lambda env: DiscreteActSpace(env.action_space, attr_to_keep=["set_bus"]),
            obs_space_creation=lambda env: BoxGymObsSpace(env.observation_space, attr_to_keep=["rho", "p_or", "gen_p", "load_p"])
        )
        self.observation_space = Box(shape=self._gym_env.observation_space.shape, low=self._gym_env.observation_space.low, high=self._gym_env.observation_space.high)
        self.action_space = Discrete(self._gym_env.action_space.n)



@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    env_config = OmegaConf.to_container(cfg.env.training_env, resolve=True)
    env_config["hydra"] = cfg

    ray_conf = cfg.ra_dqn.model.ray
    config = (
        DQNConfig()
        .environment(env=RayEnv, env_config=env_config)
        .training(
            epsilon=[[0, 1.0], [ray_conf.exploration_fraction * cfg.ra_dqn.train.timesteps, ray_conf.exploration_final_eps]],
            target_network_update_freq=ray_conf.target_network_update_freq,
            num_steps_sampled_before_learning_starts=ray_conf.num_steps_sampled_before_learning_starts,
            lr=ray_conf.lr,
            dueling=ray_conf.dueling,
            double_q=ray_conf.double_q,
            train_batch_size_per_learner=ray_conf.train_batch_size_per_learner,
            replay_buffer_config={
                "_enable_replay_buffer_api": True,
                "type": ray_conf.replay_buffer_config.type,
                "capacity": ray_conf.replay_buffer_config.capacity,
                "prioritized_replay_alpha": ray_conf.replay_buffer_config.alpha,
                "prioritized_replay_beta": ray_conf.replay_buffer_config.beta,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
            },
            td_error_loss_fn="huber"
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .env_runners(num_env_runners=2, num_envs_per_env_runner=1, num_cpus_per_env_runner=1)
        .framework("torch")
    )
    dqn = config.build()
    print(dqn.train())


if __name__ == "__main__":
    main()
