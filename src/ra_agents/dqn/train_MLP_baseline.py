import os.path
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN

from src.common.constants import set_experiment_name, logger, SEED
from .CustomDQN import SoftmaxDQN
from ..utils import get_env_mlp_baseline, evaluate, get_callbacks


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    set_experiment_name(cfg.experiment_name)
    from src.common.constants import EVAL_PATH,LOGS_PATH, MODELS_PATH
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]
    group = "baselines/dqn"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"mlp_{timestamp}"
    else:
        name = f"mlp_{timestamp}_{name_suffix}"

    # create env
    env = get_env_mlp_baseline(cfg)

    # create algorithm
    if cfg.rl.dqn.exploration == "Softmax":
        algorithm = SoftmaxDQN(
            env=env,
            policy="MlpPolicy",
            tau_start=cfg.rl.dqn.sb3.tau_start,
            tau_end=cfg.rl.dqn.sb3.tau_end,
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
            },
            seed = SEED,
        )
    else:
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
            policy_kwargs={
                "net_arch": cfg.rl.dqn.sb3.policy_kwargs.net_arch,
            },
            buffer_size=cfg.rl.dqn.sb3.buffer_size,
            batch_size=cfg.rl.dqn.sb3.batch_size,
            learning_rate=cfg.rl.dqn.sb3.learning_rate,
            seed=SEED,
        )

    # train
    path_results = Path(EVAL_PATH, group, name)
    callbacks = get_callbacks(cfg, path_results, env_creation=get_env_mlp_baseline)
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=cfg.rl.train.log_interval, callback=callbacks)
    algorithm.save(os.path.join(MODELS_PATH, group, name + ".zip"))

    # evaluate
    evaluate(algorithm, get_env_mlp_baseline, path_results, cfg)


if __name__ == "__main__":
    main()
