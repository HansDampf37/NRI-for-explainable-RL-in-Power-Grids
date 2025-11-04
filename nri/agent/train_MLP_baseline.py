import uuid
from datetime import datetime

import hydra
from grid2op.gym_compat import BoxGymObsSpace
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN

from common import G2OpGymEnv


def get_env(cfg) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment from hydra config using action and observation spaces from the configs baseline

    :param cfg: The hydra config
    :return: The environment
    """
    env: G2OpGymEnv = instantiate(
        cfg.env,
        obs_space_creation=lambda e: BoxGymObsSpace(grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.ra_dqn.act_space, grid2op_action_space=e.action_space)
    )
    return env


@hydra.main(config_path="../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    name = f"mlp_{timestamp}_{uuid.uuid4().hex}"
    env = get_env(cfg)

    algorithm = DQN(
        env=env,
        policy="MlpPolicy",
        tensorboard_log="data/logs/baseline",
        verbose=cfg.ra_dqn.model.sb3.verbose,
        train_freq=cfg.ra_dqn.model.sb3.train_freq,
        gradient_steps=cfg.ra_dqn.model.sb3.gradient_steps,
        gamma=cfg.ra_dqn.model.sb3.gamma,
        exploration_fraction=cfg.ra_dqn.model.sb3.exploration_fraction,
        exploration_final_eps=cfg.ra_dqn.model.sb3.exploration_final_eps,
        target_update_interval=cfg.ra_dqn.model.sb3.target_update_interval,
        learning_starts=cfg.ra_dqn.model.sb3.learning_starts,
        buffer_size=cfg.ra_dqn.model.sb3.buffer_size,
        batch_size=cfg.ra_dqn.model.sb3.batch_size,
        learning_rate=cfg.ra_dqn.model.sb3.learning_rate,
    )
    algorithm.learn(total_timesteps=int(1e6), tb_log_name=name, log_interval=cfg.ra_dqn.train.log_interval)


if __name__ == "__main__":
    main()
