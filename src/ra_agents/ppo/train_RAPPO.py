import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.common.constants import set_experiment_name, SEED
from src.common.observation_space import EDGE_INDEX, BusConnectivityGraphObsSpace
from src.nri.utils import prior_from_env
from src.visualization.utils import PlottingArgs, get_node_styles
from .RAPPO import RAPPO
from ..RAFeatureExtractor import RAFeatureExtractorSB3
from ..get_edge_probs import save_edge_probs
from ..pretrain_encoder import main as pretrain_encoder
from ..utils import get_env, evaluate, get_callbacks

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="configs", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.rl.verbose:
        logger.info(OmegaConf.to_yaml(cfg))

    # get paths
    set_experiment_name(cfg.experiment_name)
    from src.common.constants import EVAL_PATH, LOGS_PATH, MODELS_PATH, EDGE_PROBS_PATH
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]
    group = "relation-aware/graphormer_ppo" if cfg.rl.model.use_graphormer else "relation-aware/ppo"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"rappo_{timestamp}"
    else:
        name = f"rappo_{timestamp}_{name_suffix}"

    # create env
    env = get_env(cfg)

    # create prior distribution that edge-type predictions will be pushed towards
    prior = prior_from_env(
        prob_graph_edge_exists=cfg.rl.model.prior_for_graph_edges_existing,
        env=env,
        temperature=cfg.rl.model.temperature,
        verbose=cfg.rl.verbose
    )

    # create policy kwargs
    policy_kwargs = {
        "net_arch": cfg.rl.ppo.sb3.policy_kwargs.net_arch,
        "features_extractor_class": RAFeatureExtractorSB3,
        "features_extractor_kwargs": {
            "hidden_dim": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.hidden_dim,
            "out_dim": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.out_dim,
            "num_edge_types": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.num_edge_types,
            "num_layers": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.num_layers,
            "dropout_prob": cfg.rl.ppo.sb3.policy_kwargs.features_extractor_kwargs.dropout_prob,
            "use_graphormer": cfg.rl.model.use_graphormer,
            "max_degree": cfg.rl.model.features_extractor_kwargs.max_degree,
            "max_path_distance": cfg.rl.model.features_extractor_kwargs.max_path_distance,
        }
    }

    # create plotting args (for logging)
    plotting_args = PlottingArgs(
        num_nodes=env.observation_space.num_nodes,
        node_styles=get_node_styles(env._g2op_env, BusConnectivityGraphObsSpace),
        powerline_edge_index=torch.from_numpy(env.reset()[0][EDGE_INDEX]).cpu().numpy(),
        skip_last_edge_type=True,
    )

    # create algorithm
    algorithm = RAPPO(
        env=env,
        plotting_args=plotting_args,
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
        kl_coef=cfg.rl.ppo.sb3.kl_coef,
        max_grad_norm=cfg.rl.ppo.sb3.max_grad_norm,
        use_sde=cfg.rl.ppo.sb3.use_sde,
        sde_sample_freq=cfg.rl.ppo.sb3.sde_sample_freq,
        tensorboard_log=os.path.join(LOGS_PATH, group),
        policy_kwargs=policy_kwargs,
        seed=SEED,
    )
    algorithm.set_prior(prior)

    # pretrain and set encoder
    encoder = pretrain_encoder(cfg)
    algorithm.policy.features_extractor.gnn_feature_extractor.encoder = encoder

    # train
    path_results = Path(EVAL_PATH, group, name)
    callbacks = get_callbacks(cfg, path_results)
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=1, callback=callbacks)
    algorithm.save(os.path.join(MODELS_PATH, group, name + ".zip"))

    # evaluate and return results
    results_dict = evaluate(
        algorithm=algorithm,
        env_creation=get_env,
        path_results=path_results,
        cfg=cfg,
        verbose=cfg.rl.verbose
    )

    # save edge probs
    save_edge_probs(
        RARL_model=algorithm,
        env=env,
        save_path=Path(EDGE_PROBS_PATH, group, name + ".npy"),
        num_samples=cfg.rl.eval.final.num_samples_for_edge_average,
        verbose=cfg.rl.verbose
    )

    return results_dict


if __name__ == "__main__":
    main()
