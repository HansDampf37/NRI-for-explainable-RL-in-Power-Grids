"""
This script implements the relations aware DQN (RADQN) in the sb3 framework.
"""
from typing import Union, Optional, Tuple, Any, Dict, List

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.type_aliases import GymEnv, PyTorchObs, Schedule, MaybeCallback
from stable_baselines3.dqn.dqn import SelfDQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn, Tensor

from src.common.observation_space import GraphObservationSpace
from src.visualization.utils import visualize_graph, PlottingArgs, visualize_posterior
from .HuberKLLoss import HuberKLLoss
from .CustomDQN import SoftmaxDQN
from ..RAFeatureExtractor import RAFeatureExtractorSB3
from ..RARL import RARL


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


class RADQN(SoftmaxDQN, RARL):
    """
    This class implements the DQN interface from sb3. It uses an Encoder + downstream RA-GNN to predict the q_values.
    The loss is extended, to include the distance between posterior p(z|x) to the prior p(z).
    """

    def set_loss_function(self, loss_fn: HuberKLLoss):
        """
        @param loss_fn: a HuberKLLoss object that is used to train the DQN
        """
        self.loss_fn = loss_fn


    def __init__(self,
                 env: Union[GymEnv, str],
                 policy: type[RADQNPolicy] = RADQNPolicy,
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
                 _init_setup_model: bool = True,
                 tau_start: float = 0.5,
                 tau_end: float = 0.0000001) -> None:
        """
        Constructor.
        @param env: the environment
        """
        super().__init__(
            policy,
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
            _init_setup_model,
            tau_start=tau_start,
            tau_end=tau_end)
        assert env is None or isinstance(env.observation_space, GraphObservationSpace), "RADQN requires a graph observation space"
        self.loss_fn: Optional[HuberKLLoss] = None
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
        var_posteriors = []

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
            mean_posteriors.append(posterior_distributions.mean(dim=0).detach().cpu().numpy()) # mean over batch dim -> [E, K]
            var_posteriors.append(posterior_distributions.var(dim=0).detach().cpu().numpy()) # mean over batch dim -> [E, K]

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
        tb_formatter = next(
            (fmt for fmt in self.logger.output_formats if isinstance(fmt, TensorBoardOutputFormat)),
            None
        )
        if tb_formatter is not None:
            writer = tb_formatter.writer  # this is the SummaryWriter
            mean_posterior = np.mean(mean_posteriors, axis=0) # mean over iterations -> [E, K]
            var_posterior = np.stack(var_posteriors, axis=0)

            writer.add_histogram("latent_edges/posterior example", mean_posterior, global_step=self.num_timesteps, bins=40)
            writer.add_histogram("latent_edges/variance posterior", var_posterior[:, :, :-1], global_step=self.num_timesteps)
            posterior_hist_image = visualize_posterior(mean_posterior, self.loss_fn.prior.detach().cpu().numpy())
            writer.add_figure("latent_edges/posterior_vs_prior", posterior_hist_image, global_step=self.num_timesteps)
            if self.plotting_args is not None:
                self.plotting_args.latent_edge_probs = mean_posterior
                mean_latent_edges_image = visualize_graph(self.plotting_args)
                writer.add_figure("latent_edges/latent-edges", mean_latent_edges_image, global_step=self.num_timesteps)

    def get_edge_type_posterior(self, obs: Union[np.ndarray, dict[str, np.ndarray]]) -> Tensor:
        _, edge_type_posterior = self.q_net(self.q_net.obs_to_tensor(obs)[0])
        return edge_type_posterior


    def learn(self: SelfDQN, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 4,
              tb_log_name: str = "DQN", reset_num_timesteps: bool = True, progress_bar: bool = False) -> SelfDQN:
        assert self.loss_fn is not None
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
