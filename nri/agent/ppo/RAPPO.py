"""
This script implements the relations aware PPO (RAPPO) in the sb3 framework.
"""
from typing import Union, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, PyTorchObs
from stable_baselines3.common.utils import explained_variance
from torch import Tensor

from common import GraphObservationSpace, BusConnectivityGraphObsSpace
from visualization.utils import visualize_graph, PlottingArgs, visualize_posterior


class RAPPO(PPO):
    """
    This class implements the PPO interface from sb3. It uses an Encoder + downstream RA-GNN to predict the action probabilities + q-value.
    The loss is extended, to include the distance between posterior p(z|x) to the prior p(z).
    """

    def __init__(self,
                 env: Union[GymEnv, str],
                 prior: Tensor,
                 plotting_args: Optional[PlottingArgs] = None,
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 kl_coef: float = 1.0,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
                 rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
                 target_kl: Optional[float] = None,
                 stats_window_size: int = 100,
                 tensorboard_log: Optional[str] = None,
                 policy_kwargs: Optional[dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[torch.device, str] = "auto",
                 _init_setup_model: bool = True) -> None:
        """
        Constructor.
        @param env: the environment
        @param prior: tensor of shape [E, K] containing prior distributions that the posterior edge type distributions will be pushed towards.
        @param kl_coef: the weight of the KL term in the objective
        """
        super().__init__(
            RAPPOPolicy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model)
        assert isinstance(env.observation_space, GraphObservationSpace), "RADQN requires a graph observation space"
        self.plotting_args = plotting_args
        self.prior = prior.to(device=self.device, dtype=torch.float32)
        self.kl_coef = kl_coef
        self.eps = 1e-10
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def train(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        mean_posteriors = []
        kl_divs = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                posterior_distributions = self.policy.get_edge_type_posterior(rollout_data.observations)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                mean_posteriors.append(posterior_distributions.mean(dim=0).detach().cpu().numpy())  # mean over batch dim -> [E, K]
                last_posterior = posterior_distributions[0].detach().cpu().numpy()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                kl_loss = (posterior_distributions * (torch.log(posterior_distributions + self.eps) - torch.log(self.prior + self.eps))).sum(dim=-1)
                kl_loss = kl_loss.mean()
                kl_divs = kl_loss.item()

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.kl_coef * kl_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/kl-div", np.mean(kl_divs))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # visualize and plot images
        tb_formatter = next(
            (fmt for fmt in self.logger.output_formats if isinstance(fmt, TensorBoardOutputFormat)),
            None
        )
        if tb_formatter is not None:
            writer = tb_formatter.writer  # this is the SummaryWriter
            mean_posterior = np.mean(mean_posteriors, axis=0)  # mean over iterations -> [E, K]
            writer.add_histogram("train/posterior example", mean_posterior, global_step=self.num_timesteps)
            hist_image = visualize_posterior(mean_posterior, self.prior.detach().cpu().numpy())
            writer.add_figure("train/posterior_vs_prior", hist_image, global_step=self.num_timesteps)
            if self.plotting_args is not None:
                self.plotting_args.latent_edge_probs = np.mean(mean_posteriors, axis=0) # mean over iterations -> [E, K]
                mean_latent_edges_image = visualize_graph(self.plotting_args)
                writer.add_figure("train/latent-edges", mean_latent_edges_image, global_step=self.num_timesteps)



class RAPPOPolicy(ActorCriticPolicy):
    """
    This Policy is just like the ActorCriticPolicy with the difference that it also passes the predicted posterior edge types
    """

    def __init__(
        self,
        observation_space: BusConnectivityGraphObsSpace,
        action_space: Discrete,
        lr_schedule: Schedule,
        **kwargs,
    ):
        share = kwargs.get("share_features_extractor", True)
        if share is not True:
            raise NotImplementedError("This Policy requires share_features_extractor=True")

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs,
        )

    def forward(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features, posterior_edge_types = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        features, _ = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> Tensor:
        features, _ = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs: PyTorchObs, actions: Tensor) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        # Preprocess the observation if needed
        features, _ = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_edge_type_posterior(self, obs: PyTorchObs) -> Tensor:
        """
        RAPPO predicts type distributions for each edge of the fully connected graph such "exists" / "doesn't exist"
        @param obs: the input
        @return: edge type distributions [E, K]
        """
        _, edge_type_posterior = self.extract_features(obs)
        return edge_type_posterior
