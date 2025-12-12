import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.common.constants import set_experiment_name, SEED
from .CustomPPO import G2OpPPO
from ..RAFeatureExtractor import BaselineFeatureExtractorSB3
from ..utils import get_env, evaluate, EvalCallback, CurriculumCallback

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.rl.verbose:
        logger.info(OmegaConf.to_yaml(cfg))

    # get paths
    set_experiment_name(cfg.experiment_name)
    from src.common.constants import EVAL_PATH,LOGS_PATH, MODELS_PATH
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]
    group = "baselines/ppo"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"gnn_{timestamp}"
    else:
        name = f"gnn_{timestamp}_{name_suffix}"

    # create env
    env = get_env(cfg)

    # create policy kwargs
    policy_kwargs = {
        "net_arch": cfg.rl.ppo.sb3.policy_kwargs.net_arch,
        "features_extractor_class": BaselineFeatureExtractorSB3,
        "features_extractor_kwargs": {
            "hidden_dim": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.hidden_dim,
            "out_dim": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.out_dim,
            "num_layers": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.num_layers,
            "dropout_prob": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.dropout_prob,
        }
    }

    # create algorithm
    algorithm = G2OpPPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=cfg.rl.ppo.sb3.verbose and cfg.rl.verbose,
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
        policy_kwargs=policy_kwargs,
        seed=SEED,
    )

    # train
    path_results = Path(EVAL_PATH, group, name)
    eval_callback = EvalCallback(
        eval_freq=max(cfg.rl.train.timesteps // cfg.rl.eval.num_evaluations_during_training, 1),
        env_fn=get_env,
        path_results_root=Path(path_results, "checkpoints"),
        cfg=cfg,
        verbose=1 if cfg.rl.verbose else 0
    )
    curriculum_cb = CurriculumCallback(
        total_timesteps=cfg.rl.train.timesteps,
        level1_at_fraction=float(cfg.env.curriculum_level_config.level1_at_fraction),
        level2_at_fraction=float(cfg.env.curriculum_level_config.level2_at_fraction),
        start_level=int(cfg.env.curriculum_level_config.start_level),
        verbose=1 if cfg.rl.verbose else 0,
    )

    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=1, callback=[eval_callback, curriculum_cb])
    algorithm.save(os.path.join(MODELS_PATH, group, name + ".zip"))

    # evaluate and return results
    results_dict = evaluate(
        algorithm=algorithm,
        env_creation=get_env,
        path_results=path_results,
        cfg=cfg,
        verbose=cfg.rl.verbose
    )

    return results_dict

if __name__ == "__main__":
    main()
