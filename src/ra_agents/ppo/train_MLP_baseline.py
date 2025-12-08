import os.path
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO

from src.common.constants import set_experiment_name, logger, SEED
from .PPOTopoPolicy import Sb3PPOTopologyPolicy
from ..utils import get_env_mlp_baseline, evaluate, EvalCallback


@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    set_experiment_name(cfg.experiment_name)
    from src.common.constants import EVAL_PATH,LOGS_PATH, MODELS_PATH
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    group = "baselines/ppo"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"mlp_{timestamp}"
    else:
        name = f"mlp_{timestamp}_{name_suffix}"

    # create env
    env = get_env_mlp_baseline(cfg)

    # create algorithm
    algorithm = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=cfg.rl.ppo.sb3.verbose,
        learning_rate=cfg.rl.ppo.sb3.learning_rate,
        n_steps=cfg.rl.ppo.sb3.n_steps,
        batch_size=cfg.rl.ppo.sb3.batch_size,
        n_epochs=cfg.rl.ppo.sb3.n_epochs,
        gamma=cfg.rl.ppo.sb3.gamma,
        gae_lambda=cfg.rl.ppo.sb3.gae_lambda,
        clip_range=cfg.rl.ppo.sb3.clip_range,
        clip_range_vf=cfg.rl.ppo.sb3.clip_range_vf,
        normalize_advantage=cfg.rl.ppo.sb3.normalize_advantage,
        ent_coef=cfg.rl.ppo.sb3.ent_coef,
        vf_coef=cfg.rl.ppo.sb3.vf_coef,
        max_grad_norm=cfg.rl.ppo.sb3.max_grad_norm,
        use_sde=cfg.rl.ppo.sb3.use_sde,
        sde_sample_freq=cfg.rl.ppo.sb3.sde_sample_freq,
        tensorboard_log=os.path.join(LOGS_PATH, group),
        policy_kwargs={
            "net_arch": cfg.rl.ppo.sb3.policy_kwargs.net_arch,
        },
        seed=SEED
    )

    # train
    path_results = Path(EVAL_PATH, group, name)
    topology_policy = Sb3PPOTopologyPolicy(algorithm)
    eval_callback = EvalCallback(
        eval_freq=max(cfg.rl.train.timesteps // 10, 1), # evaluate model 10 times during training
        env_fn=get_env_mlp_baseline,
        path_results_root=Path(path_results, "checkpoints"),
        cfg=cfg,
        topology_policy=topology_policy
    )
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=1, callback=eval_callback)
    algorithm.save(os.path.join(MODELS_PATH, group, name + ".zip"))

    # evaluate
    evaluate(algorithm, topology_policy, get_env_mlp_baseline, path_results, cfg)


if __name__ == "__main__":
    main()
