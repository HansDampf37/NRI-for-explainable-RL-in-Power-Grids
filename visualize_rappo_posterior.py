"""
Simplified script to run a saved RAPPO checkpoint and display the posterior distribution.

This version shows posterior statistics and graph visualizations.
Reuses existing infrastructure from evaluate_rllib_agent.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from ray.rllib.models import ModelCatalog

from src.common.observation_space import BusConnectivityGraphObsSpace
from src.ra_agents.RAFeatureExtractor import RLlibGNNModel, RLlibRAGNNModel, RLlibNRIGNNModel
from src.visualization import get_node_styles
# Reuse existing evaluation infrastructure
from evaluate_rllib_agent import load_env_config_from_params, load_rllib_agent
from src.rl4pnc.evaluation.evaluation_agents import RllibAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RAPPOPosteriorAgent(BaseAgent):
    """Wrapper around RllibAgent that displays RAPPO posterior distribution when RL model acts."""

    def __init__(self, rllib_agent: RllibAgent, gym_wrapper, visualize_every_n: int = 1):
        BaseAgent.__init__(self, rllib_agent.action_space)

        self.rllib_agent = rllib_agent
        self.gym_wrapper = gym_wrapper
        self.visualize_every_n = visualize_every_n
        self.step_count = 0
        self.rl_action_count = 0  # Count of times RL model actually computed an action

        # Get the policy model for accessing posterior
        self.policy_model = rllib_agent._rllib_agent.model

        # Get environment and node styles for graph visualization
        self.g2op_env = gym_wrapper.env_gym.init_env
        self.node_styles = get_node_styles(self.g2op_env, BusConnectivityGraphObsSpace)

        # Track chronic ID for saving files
        self.chronic_id = None

        # For visualization - disable interactive mode to prevent empty windows
        plt.ioff()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(20, 8))

    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        """Returns action and visualizes posterior when RL model is used."""
        self.step_count += 1
        action = self.rllib_agent.act(observation, reward, done)

        # Check if RL agent will be activated (same logic as RllibAgent)
        use_rl_component = self.rllib_agent.activate_agent(observation)
        if use_rl_component:
            self.rl_action_count += 1
            print(f"\n{'='*80}")
            print(f"Step {self.step_count}: RL AGENT ACTIVATED (ρ={self.rllib_agent.rho_max:.3f} > {self.rllib_agent.activation_thresh:.3f})")
            print(f"{'='*80}")

            if self.rl_action_count % self.visualize_every_n == 0:
                self._visualize_posterior(observation)

        return action

    def _visualize_posterior(self, observation: BaseObservation):
        """Display posterior statistics and histogram."""
        posterior = self.policy_model.get_posterior()  # [B, E, K]

        if posterior.dim() == 3:
            posterior = posterior[0]  # [E, K]

        posterior_np = posterior.cpu().detach().numpy()
        edge_probs = posterior_np[:, 0]  # Edge existence probabilities

        rho_max = observation.rho.max()

        # Print statistics
        print(f"\n{'='*80}")
        print(f"RL Action #{self.rl_action_count} (Total Step {self.step_count}) | Max ρ: {rho_max:.3f}")
        print(f"{'='*80}")
        print(f"Posterior shape: {posterior_np.shape} (Edges × Edge Types)")
        print(f"\nEdge Existence Probability Statistics:")
        print(f"  Mean:   {edge_probs.mean():.4f}")
        print(f"  Std:    {edge_probs.std():.4f}")
        print(f"  Min:    {edge_probs.min():.4f}")
        print(f"  Max:    {edge_probs.max():.4f}")
        print(f"  Median: {np.median(edge_probs):.4f}")

        # Show percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = np.percentile(edge_probs, p)
            print(f"  {p:2d}th: {val:.4f}")

        # Count high-probability edges
        thresholds = [0.5, 0.7, 0.9, 0.95]
        print(f"\nEdges above threshold:")
        for thresh in thresholds:
            count = (edge_probs > thresh).sum()
            pct = 100 * count / len(edge_probs)
            print(f"  > {thresh:.2f}: {count:4d} ({pct:5.2f}%)")

        # Top edges
        print(f"\nTop 10 most confident edges:")
        top_idx = np.argsort(edge_probs)[-10:][::-1]
        for i, idx in enumerate(top_idx, 1):
            print(f"  {i:2d}. Edge {idx:4d}: {edge_probs[idx]:.4f}")

        # Update plot
        for ax in self.axes:
            ax.clear()

        # Left: Histogram
        self.axes[0].hist(edge_probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        self.axes[0].axvline(edge_probs.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {edge_probs.mean():.3f}')
        self.axes[0].axvline(np.median(edge_probs), color='green', linestyle='--', linewidth=2,
                           label=f'Median: {np.median(edge_probs):.3f}')
        self.axes[0].set_xlabel('Edge Existence Probability')
        self.axes[0].set_ylabel('Count')
        self.axes[0].set_title(f'Posterior Distribution (RL Action #{self.rl_action_count})')
        self.axes[0].legend()
        self.axes[0].grid(alpha=0.3)

        # Get power grid edges from current observation
        from src.common.observation_space import EDGE_INDEX, EDGE_MASK
        from src.visualization import visualize_graph, PlottingArgs

        gym_obs = self.gym_wrapper.cur_gym_obs
        powerline_edges = gym_obs[EDGE_INDEX][:, gym_obs[EDGE_MASK]]

        # Right: Draw graph directly into subplot using visualize_graph
        self.axes[1].clear()
        visualize_graph(PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=posterior_np,
            skip_last_edge_type=True
        ), ax=self.axes[1])

        self.axes[1].set_title('Latent Graph Structure\n(Black dashed = power grid, Colored = latent)')

        # Save
        save_dir = Path("results/visualizations/posterior_simple")
        save_dir.mkdir(parents=True, exist_ok=True)
        chronic_name = self.chronic_id if self.chronic_id else "unknown"

        self.fig.suptitle(f'RAPPO Posterior - RL Action #{self.rl_action_count}, Step {self.step_count} (Max ρ: {rho_max:.3f})',
                        fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save combined figure
        save_path = save_dir / f"{chronic_name}_rl_action_{self.rl_action_count:04d}_step_{self.step_count:04d}.png"
        self.fig.savefig(save_path, dpi=120, bbox_inches='tight')

        print(f"\nSaved to: {save_path}")



def run_rappo_visualization(
    checkpoint_path: str,
    policy_name: str = "reinforcement_learning_policy",
    checkpoint_name: str = "checkpoint_000010",
    env_name_override: str = None,
    num_episodes: int = 1,
    visualize_every_n: int = 5,
):
    """Run RAPPO checkpoint with posterior visualization."""

    # Register models (required before loading checkpoint)
    ModelCatalog.register_custom_model("gnn_model", RLlibGNNModel)
    ModelCatalog.register_custom_model("ragnn_model", RLlibRAGNNModel)
    ModelCatalog.register_custom_model("nrignn_model", RLlibNRIGNNModel)

    # Load config using existing function
    env_config = load_env_config_from_params(checkpoint_path)
    if env_name_override:
        env_config["env_name"] = env_name_override

    env_name = env_config["env_name"]

    logger.info(f"Loading RAPPO from: {checkpoint_path}")
    logger.info(f"Checkpoint: {checkpoint_name}")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Visualizing every {visualize_every_n} steps\n")

    # Use existing load_rllib_agent function
    rllib_agent, g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=checkpoint_path,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
        env_name=env_name,
        env_config=env_config
    )

    # Wrap with visualization
    agent = RAPPOPosteriorAgent(
        rllib_agent=rllib_agent,
        gym_wrapper=gym_wrapper,
        visualize_every_n=visualize_every_n,
    )

    logger.info("Agent loaded! Starting episodes...\n")

    # Run episodes
    for episode in range(num_episodes):
        obs = g2op_env.reset()
        done = False
        total_reward = 0
        step = 0

        # Reset counters and get chronic ID for this episode
        agent.rl_action_count = 0
        agent.step_count = 0
        agent.chronic_id = g2op_env.chronics_handler.get_name()

        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}/{num_episodes} - Chronic: {agent.chronic_id}")
        print(f"{'='*80}\n")

        while not done:
            action = agent.act(obs, total_reward, done)
            obs, reward, done, info = g2op_env.step(action)
            total_reward += reward
            step += 1

            if done:
                print(f"\n{'='*80}")
                print(f"Episode {episode + 1} finished!")
                print(f"  Total steps: {step}")
                print(f"  RL actions:  {agent.rl_action_count} ({100*agent.rl_action_count/step:.1f}%)")
                print(f"  Heuristic actions: {step - agent.rl_action_count} ({100*(step-agent.rl_action_count)/step:.1f}%)")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Reason: {info.get('exception', 'Completed')}")
                print(f"{'='*80}\n")

    logger.info("Visualization complete!")
    logger.info(f"Saved to: results/visualizations/posterior_simple/")


    # Cleanup - reuse try/except pattern from existing code
    try:
        gym_wrapper.env_gym.close()
    except Exception as e:
        logger.debug(f"Environment cleanup error (ignored): {e}")


def main():
    """Main entry point."""

    # CONFIGURATION - Edit these parameters
    checkpoint_path = "/home/adrian/Schreibtisch/1501_rappo_with_anneal/CustomPPO_0_a0abf_2026-01-15_11-54-55/"
    checkpoint_name = "checkpoint_000020"
    env_name_override = "l2rpn_case14_sandbox_val"  # Use None to auto-load from params.json

    num_episodes = 2  # Number of episodes to run
    # NOTE: visualize_every_n means "every N times the RL MODEL acts" (not every N steps)
    # Since heuristics handle most actions, the RL model only acts when rho > threshold
    visualize_every_n = 1  # Visualize every time RL model acts (1 = all, 2 = every 2nd, etc.)


    run_rappo_visualization(
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
        env_name_override=env_name_override,
        num_episodes=num_episodes,
        visualize_every_n=visualize_every_n,
    )


if __name__ == "__main__":
    main()

