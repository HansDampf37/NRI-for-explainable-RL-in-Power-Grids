import os.path
import uuid
from datetime import datetime

import hydra
from grid2op.gym_compat import BoxGymObsSpace
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN

from baselines.baseline_agent import evaluate_topology_policy
from common import G2OpGymEnv
from common.constants import LOGS_PATH, MODELS_PATH
from nri.agent.dqn.DQNTopoPolicy import Sb3DQNTopologyPolicy


def get_env(cfg) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action space from hydra config. Observation space is always
    BoxGymObsSpace since we will use an MLP instead of a GNN

    :param cfg: The hydra config
    :return: The environment
    """
    env: G2OpGymEnv = instantiate(
        cfg.env.training_env,
        obs_space_creation=lambda e: BoxGymObsSpace(grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space)
    )
    return env


@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    group = "rl/relation-unaware-baselines/dqn"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"mlp_{timestamp}_{uuid.uuid4().hex}"
    else:
        name = f"mlp_{timestamp}_{name_suffix}_{uuid.uuid4().hex}"

    # create env
    env = get_env(cfg)

    # create algorithm
    algorithm = DQN(
        env=env,
        policy="MlpPolicy",
        tensorboard_log=os.path.join(LOGS_PATH, group),
        verbose=cfg.rl.dqn.sb3.verbose,
        train_freq=cfg.rl.dqn.sb3.train_freq,
        gradient_steps=cfg.rl.dqn.sb3.gradient_steps,
        gamma=cfg.rl.dqn.sb3.gamma,
        exploration_fraction=cfg.rl.dqn.sb3.exploration_fraction,
        exploration_final_eps=cfg.rl.dqn.sb3.exploration_final_eps,
        target_update_interval=cfg.rl.dqn.sb3.target_update_interval,
        learning_starts=cfg.rl.dqn.sb3.learning_starts,
        buffer_size=cfg.rl.dqn.sb3.buffer_size,
        batch_size=cfg.rl.dqn.sb3.batch_size,
        learning_rate=cfg.rl.dqn.sb3.learning_rate,
        policy_kwargs={
            "net_arch": cfg.rl.dqn.sb3.policy_kwargs.net_arch,
        }
    )

    # train
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=cfg.rl.train.log_interval)
    algorithm.save(os.path.join(MODELS_PATH, group, name))
    topology_policy = Sb3DQNTopologyPolicy(algorithm)

    # evaluate
    evaluate_topology_policy(topology_policy, group, name, cfg)


if __name__ == "__main__":
    main()
