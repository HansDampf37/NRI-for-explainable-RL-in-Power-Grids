import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.common.constants import set_experiment_name, logger, SEED
from src.common.observation_space import EDGE_INDEX, BusConnectivityGraphObsSpace
from src.nri.utils import prior_from_env
from src.visualization.utils import PlottingArgs, get_node_styles
from .HuberKLLoss import HuberKLLoss
from .RADQN import RADQN
from ..RAFeatureExtractor import RAFeatureExtractorSB3
from ..get_edge_probs import save_edge_probs
from ..pretrain_encoder import main as pretrain_encoder
from ..utils import get_env, evaluate, get_callbacks


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    set_experiment_name(cfg.experiment_name)
    from src.common.constants import EVAL_PATH, LOGS_PATH, MODELS_PATH, EDGE_PROBS_PATH
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]
    group = "relation-aware/dqn"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"radqn_{timestamp}"
    else:
        name = f"radqn_{timestamp}_{name_suffix}"

    # create env
    env = get_env(cfg)

    if cfg.rl.model.use_graphormer:
        group += "/graphormer"

    # create policy kwargs
    policy_kwargs = {
        "net_arch": cfg.rl.dqn.sb3.policy_kwargs.net_arch,
        "features_extractor_class": RAFeatureExtractorSB3,
        "features_extractor_kwargs": {
            "hidden_dim": cfg.rl.dqn.sb3.policy_kwargs.features_extractor_kwargs.hidden_dim,
            "out_dim": cfg.rl.dqn.sb3.policy_kwargs.features_extractor_kwargs.out_dim,
            "num_edge_types": cfg.rl.dqn.sb3.policy_kwargs.features_extractor_kwargs.num_edge_types,
            "num_layers": cfg.rl.dqn.sb3.policy_kwargs.features_extractor_kwargs.num_layers,
            "dropout_prob": cfg.rl.dqn.sb3.policy_kwargs.features_extractor_kwargs.dropout_prob,
            "use_graphormer": cfg.rl.model.use_graphormer,
            "max_degree": cfg.rl.model.features_extractor_kwargs.max_degree,
            "max_path_distance": cfg.rl.model.features_extractor_kwargs.max_path_distance,
            # TODO optionally include edge index here to restrict edges for nri
        }
    }

    # create loss function
    prior = prior_from_env(cfg.rl.model.prior_for_graph_edges_existing, env, cfg.rl.model.temperature)
    loss_fn = HuberKLLoss(prior=prior, alpha=cfg.rl.dqn.loss.alpha, beta=cfg.rl.dqn.loss.beta)

    # create plotting args (for logging)
    plotting_args = PlottingArgs(
        num_nodes=env.observation_space.num_nodes,
        node_styles=get_node_styles(env._g2op_env, BusConnectivityGraphObsSpace),
        powerline_edge_index=torch.from_numpy(env.reset()[0][EDGE_INDEX]).cpu().numpy(),
        skip_last_edge_type=True,
    )

    # create algorithm
    algorithm = RADQN(
        env=env,
        tau_start=cfg.rl.dqn.sb3.tau_start,
        tau_end=cfg.rl.dqn.sb3.tau_end,
        tensorboard_log=os.path.join(LOGS_PATH, group),
        plotting_args=plotting_args,
        policy_kwargs=policy_kwargs,
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
        seed=SEED,
    )
    algorithm.set_loss_function(loss_fn)

    # pretrain and set encoder
    encoder1 = pretrain_encoder(cfg)
    encoder2 = pretrain_encoder(cfg)
    algorithm.q_net.features_extractor.gnn_feature_extractor.encoder = encoder1
    algorithm.q_net_target.features_extractor.gnn_feature_extractor.encoder = encoder2

    # train
    path_results = Path(EVAL_PATH, group, name)
    callbacks = get_callbacks(cfg, path_results)
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=cfg.rl.train.log_interval, callback=callbacks)
    algorithm.save(os.path.join(MODELS_PATH, group, name + ".zip"))

    # evaluate
    evaluate(algorithm, get_env, path_results, cfg)

    # save edge probs
    save_edge_probs(
        RARL_model=algorithm,
        env=env,
        save_path=Path(EDGE_PROBS_PATH, group, name + ".npy"),
        num_samples=cfg.rl.eval.final.num_samples_for_edge_average,
    )


if __name__ == "__main__":
    main()
