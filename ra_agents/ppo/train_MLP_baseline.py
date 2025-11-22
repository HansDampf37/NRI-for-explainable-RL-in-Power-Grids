import os.path
import uuid
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO

from common.baseline_agent import evaluate_topology_policy, evaluate_sb3_alg
from common.constants import LOGS_PATH, MODELS_PATH
from agent.ppo.PPOTopoPolicy import Sb3PPOTopologyPolicy
from agent.utils import get_env_mlp_baseline


@hydra.main(config_path="../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    group = "rl/relation-unaware-baselines/ppo"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"mlp_{timestamp}_{uuid.uuid4().hex}"
    else:
        name = f"mlp_{timestamp}_{name_suffix}_{uuid.uuid4().hex}"

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
        }
    )

    # train
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=1)
    algorithm.save(os.path.join(MODELS_PATH, group, name))
    topology_policy = Sb3PPOTopologyPolicy(algorithm)

    # evaluate
    evaluate_topology_policy(topology_policy, group, name, cfg)
    for dataset in ["train", "test", "val"]:
        env_dataset = get_env_mlp_baseline(cfg, f"l2rpn_case14_sandbox_{dataset}")
        evaluate_sb3_alg(algorithm, env_dataset, group, name, dataset, cfg)


if __name__ == "__main__":
    main()
