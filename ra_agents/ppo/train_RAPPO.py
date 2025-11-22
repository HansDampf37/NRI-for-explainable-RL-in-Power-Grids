import os
import uuid
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from common.baseline_agent import evaluate_topology_policy, evaluate_sb3_alg
from common import EDGE_INDEX, BusConnectivityGraphObsSpace
from common.constants import LOGS_PATH, MODELS_PATH, EDGE_PROBS_PATH
from agent.RAFeatureExtractor import RAFeatureExtractorSB3
from agent.get_edge_probs import save_edge_probs
from agent.ppo.PPOTopoPolicy import Sb3PPOTopologyPolicy
from agent.ppo.RAPPO import RAPPO
from agent.pretrain_encoder import main as pretrain_encoder
from agent.utils import get_env
from nri.utils import prior_from_env
from visualization.utils import PlottingArgs, get_node_styles


@hydra.main(config_path="../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    group = "rl/relation-aware/ppo"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"rappo_{timestamp}_{uuid.uuid4().hex}"
    else:
        name = f"rappo_{timestamp}_{name_suffix}_{uuid.uuid4().hex}"

    # create env
    env = get_env(cfg)

    if cfg.rl.model.use_graphormer:
        group += "/graphormer"

    # create prior distribution that edge-type predictions will be pushed towards
    prior = prior_from_env(cfg.rl.model.prior_for_graph_edges_existing, env, cfg.rl.model.temperature)

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
            # TODO optionally include edge index here to restrict edges for nri
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
        plotting_args=plotting_args,
        prior=prior,
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
        policy_kwargs=policy_kwargs,
    )

    # pretrain and set encoder
    graphormer_encoder = pretrain_encoder(cfg)
    algorithm.policy.features_extractor.gnn_feature_extractor.encoder = graphormer_encoder

    # train
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=1)
    algorithm.save(os.path.join(MODELS_PATH, group, name))
    topology_policy = Sb3PPOTopologyPolicy(algorithm)

    # evaluate
    evaluate_topology_policy(topology_policy, group, name, cfg)
    for dataset in ["train", "test", "val"]:
        env_dataset = get_env(cfg, f"l2rpn_case14_sandbox_{dataset}")
        evaluate_sb3_alg(algorithm, env_dataset, group, name, dataset, cfg)

    # save edge probs
    save_edge_probs(algorithm, env, save_path=Path(EDGE_PROBS_PATH, group, name + ".npy"))


if __name__ == "__main__":
    main()
