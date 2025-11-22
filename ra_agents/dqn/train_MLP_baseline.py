import os.path
import uuid
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN

from common.constants import LOGS_PATH, MODELS_PATH, EVAL_PATH
from .DQNTopoPolicy import Sb3DQNTopologyPolicy
from ..utils import get_env_mlp_baseline, evaluate


@hydra.main(config_path="../../hydra_configs", config_name="config", version_base="1.3")
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
    env = get_env_mlp_baseline(cfg)

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

    # evaluate
    topology_policy = Sb3DQNTopologyPolicy(algorithm)
    path_results = Path(EVAL_PATH, group, name)
    evaluate(algorithm, topology_policy, get_env_mlp_baseline, path_results, cfg)


if __name__ == "__main__":
    main()
