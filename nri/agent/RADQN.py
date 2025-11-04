"""
This script implements the relations aware DQN (RADQN) in the sb3 framework.
"""
import uuid
from datetime import datetime
from typing import Union, Optional, Tuple, Any, Dict, List

import hydra
import numpy as np
import torch
from gymnasium import spaces
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.type_aliases import GymEnv, PyTorchObs, Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn, Tensor

from common import GraphObservationSpace, G2OpGymEnv, EDGE_INDEX, BusConnectivityGraphObsSpace
from nri.agent.HuberKLLoss import HuberKLLoss
from nri.agent.RA_FE import RAFeatureExtractorSB3
from nri.utils import fully_connected_edge_index, _get_prior
from visualization.utils import visualize_graph, PlottingArgs, get_node_styles


class RADQN(DQN):
    """
    This class implements the DQN interface from sb3. It uses the RA-GNN to predict the q_values. The loss is extended,
    to include the distance between posterior p(z|x) to the prior p(z).
    """

    def __init__(self,
                 env: Union[GymEnv, str],
                 loss_fn: HuberKLLoss,
                 plotting_args: Optional[PlottingArgs] = None,
                 learning_rate: Union[float, Schedule] = 1e-4,
                 buffer_size: int = 1_000_000,  # 1e6
                 learning_starts: int = 100,
                 batch_size: int = 32,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 train_freq: Union[int, Tuple[int, str]] = 4,
                 gradient_steps: int = 1,
                 replay_buffer_class: Optional[type[ReplayBuffer]] = None,
                 replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
                 optimize_memory_usage: bool = False,
                 n_steps: int = 1,
                 target_update_interval: int = 10000,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 max_grad_norm: float = 10,
                 stats_window_size: int = 100,
                 tensorboard_log: Optional[str] = None,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[torch.device, str] = "auto",
                 _init_setup_model: bool = True) -> None:
        """
        Constructor.
        @param env: the environment
        @param loss_fn: a HuberKLLoss object that is used to train the DQN
        """
        super().__init__(
            RADQNPolicy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            n_steps,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model)
        assert isinstance(env.observation_space, GraphObservationSpace), "RADQN requires a graph observation space"
        self.loss_fn = loss_fn
        self.plotting_args = plotting_args
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        huber_losses = []
        kl_divs = []
        mean_posteriors = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values, _ = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates
            current_q_values, posterior_distributions = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss, huber, kl = self.loss_fn(
                current_q_values,
                target_q_values,
                posterior_distributions,
                with_huber_kl=True
            )
            losses.append(loss.item())
            huber_losses.append(huber.item())
            kl_divs.append(kl.item())
            mean_posteriors.append(posterior_distributions.mean(dim=0).detach().cpu().numpy())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/huber-loss", np.mean(huber_losses))
        self.logger.record("train/kl-div", np.mean(kl_divs))

        # visualize and plot images
        if self.plotting_args is not None:
            self.plotting_args.latent_edge_probs = np.mean(mean_posteriors, axis=0)
            mean_latent_edges_image = visualize_graph(self.plotting_args)
            tb_formatter = next(
                (fmt for fmt in self.logger.output_formats if isinstance(fmt, TensorBoardOutputFormat)),
                None
            )
            if tb_formatter is not None:
                writer = tb_formatter.writer  # this is the SummaryWriter
                writer.add_figure("train/image", mean_latent_edges_image, global_step=self._total_timesteps)


class RAQNetwork(QNetwork):
    """
    The RA-QNetwork is like the regular QNetwork with the following differences:
    1. Its Observation space is a GraphObservationSpace
    2. It uses a RA-FeatureExtractor
    3. Its forward-method outputs the predicted q-values AND the posterior distributions predicted by the FeatureExtractor
    """
    def __init__(
            self,
            observation_space: GraphObservationSpace,
            action_space: spaces.Discrete,
            features_extractor: RAFeatureExtractorSB3,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images
        )

    def forward(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the q-values.

        @param obs: Observation
        @return: The estimated Q-Value for each action, the posterior distribution predicted by the RA-FeatureExtractor.
        """
        x, p_x_given_z = self.extract_features(obs, self.features_extractor)
        return self.q_net(x), p_x_given_z

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> Tensor:
        """
        In order to make a prediction we just need the q-values and ignore the posterior
        """
        q_values, _ = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class RADQNPolicy(DQNPolicy):
    """
    This Policy is just like the DQNPolicy with the difference that it uses RA-QNetworks instead of regular QNetworks.
    """
    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return RAQNetwork(**net_args).to(self.device)


def get_env(cfg):
    """
    Creates a Grid2opWrapperEnvironment from hydra config using action and observation spaces from the configs baseline

    :param cfg: The hydra config
    :return: The environment
    """
    env: G2OpGymEnv = instantiate(
        cfg.env,
        obs_space_creation=lambda e: instantiate(cfg.ra_dqn.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.ra_dqn.act_space, grid2op_action_space=e.action_space)
    )
    return env


@hydra.main(config_path="../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    name = f"ra_dqn_{timestamp}_{uuid.uuid4().hex}"
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
        tensorboard_log="data/logs/radqn",
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


if __name__ == "__main__":
    main()
