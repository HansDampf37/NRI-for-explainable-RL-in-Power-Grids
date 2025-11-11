import os
import uuid
from datetime import datetime
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from baselines.baseline_agent import evaluate_topology_policy
from common import G2OpGymEnv, EDGE_INDEX, BusConnectivityGraphObsSpace
from common.constants import LOGS_PATH, MODELS_PATH, EDGE_PROBS_PATH
from nri.agent.RAFeatureExtractor import RAFeatureExtractorSB3
from nri.agent.dqn.DQNTopoPolicy import Sb3DQNTopologyPolicy
from nri.agent.dqn.HuberKLLoss import HuberKLLoss
from nri.agent.dqn.RADQN import RADQN
from nri.agent.get_edge_probs import save_edge_probs
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
    group = "rl/relation-aware/dqn"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"radqn_{timestamp}_{uuid.uuid4().hex}"
    else:
        name = f"radqn_{timestamp}_{name_suffix}_{uuid.uuid4().hex}"

    # create env
    env = get_env(cfg)

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
        tensorboard_log=os.path.join(LOGS_PATH, group),
        plotting_args=plotting_args,
        loss_fn=loss_fn,
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
    )

    # train
    algorithm.learn(total_timesteps=cfg.rl.train.timesteps, tb_log_name=name, log_interval=cfg.rl.train.log_interval)
    algorithm.save(os.path.join(MODELS_PATH, group, name))
    topology_policy = Sb3DQNTopologyPolicy(algorithm)

    # evaluate
    evaluate_topology_policy(topology_policy, group, name, cfg)

    # save edge probs
    save_edge_probs(algorithm, env, save_path=Path(EDGE_PROBS_PATH, group, name + ".npy"))


if __name__ == "__main__":
    main()
