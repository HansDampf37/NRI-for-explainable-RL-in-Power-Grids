import os
import uuid
from datetime import datetime

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from baselines.baseline_agent import evaluate_topology_policy
from common import G2OpGymEnv, EDGE_INDEX, BusConnectivityGraphObsSpace
from common.constants import LOGS_PATH, MODELS_PATH
from nri.agent.RAFeatureExtractor import RAFeatureExtractorSB3
from nri.agent.ppo.PPOTopoPolicy import Sb3PPOTopologyPolicy
from nri.agent.ppo.RAPPO import RAPPO
from nri.utils import prior_from_env
from visualization.utils import PlottingArgs, get_node_styles


def get_env(cfg) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action and observation spaces from hydra config.

    :param cfg: The hydra config
    :return: The environment
    """
    env: G2OpGymEnv = instantiate(
        cfg.env.training_env,
        obs_space_creation=lambda e: instantiate(cfg.rl.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space)
    )
    return env


@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    group = "rl/relation-aware/ppo"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"radqn_{timestamp}_{uuid.uuid4().hex}"
    else:
        name = f"radqn_{timestamp}_{name_suffix}_{uuid.uuid4().hex}"

    # create env
    env = get_env(cfg)

    # create prior distribution that edge-type predictions will be pushed towards
    prior = prior_from_env(cfg.rl.model.prior_for_graph_edges_existing, env)

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

    # train
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=1)
    algorithm.save(os.path.join(MODELS_PATH, group, name))
    topology_policy = Sb3PPOTopologyPolicy(algorithm)

    # evaluate
    evaluate_topology_policy(topology_policy, group, name, cfg)


if __name__ == "__main__":
    main()
