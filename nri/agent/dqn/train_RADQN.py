import os
import uuid
from datetime import datetime

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from common import G2OpGymEnv, EDGE_INDEX, BusConnectivityGraphObsSpace
from common.constants import LOGS_PATH
from nri.agent.HuberKLLoss import HuberKLLoss
from nri.agent.dqn.RADQN import RADQN
from nri.agent.RAFeatureExtractor import RAFeatureExtractorSB3
from nri.utils import fully_connected_edge_index, _get_prior
from visualization.utils import PlottingArgs, get_node_styles


def get_env(cfg) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action and observation spaces from hydra config.

    :param cfg: The hydra config
    :return: The environment
    """
    env: G2OpGymEnv = instantiate(
        cfg.env.training_env,
        obs_space_creation=lambda e: instantiate(cfg.ra_dqn.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.ra_dqn.act_space, grid2op_action_space=e.action_space)
    )
    return env


@hydra.main(config_path="../../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    group = "relations-aware"
    name = f"radqn_{timestamp}_{uuid.uuid4().hex}"
    env = get_env(cfg)

    policy_kwargs = {
        "net_arch": cfg.ra_dqn.model.sb3.policy_kwargs.net_arch,
        "features_extractor_class": RAFeatureExtractorSB3,
        "features_extractor_kwargs": {
            "hidden_dim": cfg.ra_dqn.model.sb3.policy_kwargs.features_extractor_kwargs.hidden_dim,
            "out_dim": cfg.ra_dqn.model.sb3.policy_kwargs.features_extractor_kwargs.out_dim,
            "num_edge_types": cfg.ra_dqn.model.sb3.policy_kwargs.features_extractor_kwargs.num_edge_types,
            "dropout_prob": cfg.ra_dqn.model.sb3.policy_kwargs.features_extractor_kwargs.dropout_prob, #TODO optionally include edge index here to restrict edges for nri
        }
    }

    # create loss function
    prior_for_graph_edges = Tensor(cfg.ra_dqn.model.loss.prior_for_graph_edges).to(dtype=torch.float32)
    prior_for_non_graph_edges = Tensor(cfg.ra_dqn.model.loss.prior_for_non_graph_edges).to(dtype=torch.float32)
    powergrid_edge_index = torch.from_numpy(env.reset()[0][EDGE_INDEX]) # [2, E]
    N = powergrid_edge_index.max().item() + 1
    all_edges = fully_connected_edge_index(N) # [2, E']
    prior = _get_prior(powergrid_edge_index, all_edges, prior_for_graph_edges, prior_for_non_graph_edges)
    loss_fn = HuberKLLoss(prior=prior, alpha=cfg.ra_dqn.model.loss.alpha, beta=cfg.ra_dqn.model.loss.beta)

    # create plotting args
    plotting_args = PlottingArgs(
        N,
        get_node_styles(env._g2op_env, BusConnectivityGraphObsSpace),
        powerline_edge_index=powergrid_edge_index.cpu().numpy(),
        skip_last_edge_type=True,
    )

    algorithm = RADQN(
        env=env,
        tensorboard_log=os.path.join(LOGS_PATH, group),
        plotting_args=plotting_args,
        loss_fn=loss_fn,
        policy_kwargs=policy_kwargs,
        verbose=cfg.ra_dqn.model.sb3.verbose,
        train_freq=cfg.ra_dqn.model.sb3.train_freq,
        gradient_steps=cfg.ra_dqn.model.sb3.gradient_steps,
        gamma=cfg.ra_dqn.model.sb3.gamma,
        exploration_fraction=cfg.ra_dqn.model.sb3.exploration_fraction,
        exploration_final_eps=cfg.ra_dqn.model.sb3.exploration_final_eps,
        target_update_interval=cfg.ra_dqn.model.sb3.target_update_interval,
        learning_starts=cfg.ra_dqn.model.sb3.learning_starts,
        buffer_size=cfg.ra_dqn.model.sb3.buffer_size,
        batch_size=cfg.ra_dqn.model.sb3.batch_size,
        learning_rate=cfg.ra_dqn.model.sb3.learning_rate,
    )
    algorithm.learn(total_timesteps=int(1e6), tb_log_name=name, log_interval=cfg.ra_dqn.train.log_interval)
    algorithm.save(os.path.join(LOGS_PATH, group, name))


if __name__ == "__main__":
    main()
