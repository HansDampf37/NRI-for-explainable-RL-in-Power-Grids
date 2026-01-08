"""
Script to load and evaluate an RLlib agent on Grid2Op environment.
Displays results in a boxplot similar to visualize_agent_eval_results.ipynb

IMPORTANT: This script loads a trained RLlib policy and evaluates it.
The key challenge is that RLlib's Policy.from_checkpoint() needs the correct
observation space structure. For multi-agent environments, the policy was trained
with a specific observation space (e.g., BusConnectivityGraphObsSpace), but it's
wrapped in a multi-agent Dict space during training.

When loading for evaluation, we need to provide the exact observation space structure
that the policy expects.
"""
import logging
from pathlib import Path

from ray.rllib.models import ModelCatalog

from src.common.baseline_agent import evaluate_agent
from src.common.constants import SEED
from src.ra_agents.RAFeatureExtractor import RLlibGNNModel, RLlibRAGNNModel
from src.rl4pnc.evaluation.evaluation_agents import RllibAgent
from src.rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from src.visualization import get_evaluation_metrics, visualize_agent_survival

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_gym_wrapper_from_config(env_config: dict):
    """
    Create a CustomizedGrid2OpEnvironment and return its gym wrapper.

    :param env_config: Environment configuration dictionary
    :return: GymEnv instance from CustomizedGrid2OpEnvironment
    """
    # Create the custom environment which includes the properly configured gym wrapper
    custom_env = CustomizedGrid2OpEnvironment(env_config)
    return custom_env.env_gym


def load_rllib_agent(
    checkpoint_path: str,
    policy_name: str,
    checkpoint_name: str,
    env_name: str,
    env_config: dict
):
    """
    Load an RLlib agent from a checkpoint.

    :param checkpoint_path: Path to the experiment directory containing checkpoints
    :param policy_name: Name of the policy (e.g., "reinforcement_learning_policy")
    :param checkpoint_name: Name of the checkpoint folder (e.g., "checkpoint_000000")
    :param env_name: Name of the Grid2Op environment to evaluate on
    :param env_config: Environment configuration dictionary
    :return: Tuple of (RllibAgent, Grid2Op Environment)
    """
    # Add env_name to config for CustomizedGrid2OpEnvironment
    env_config["env_name"] = env_name

    # Create the CustomizedGrid2OpEnvironment to get proper observation/action spaces
    gym_wrapper = CustomizedGrid2OpEnvironment(env_config)

    # Get the underlying Grid2Op environment for evaluation
    g2op_env = gym_wrapper.env_gym.init_env

    # Load the RLlib agent
    agent = RllibAgent(
        action_space=g2op_env.action_space,
        env_config=env_config,
        file_path=checkpoint_path,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
        gym_wrapper=gym_wrapper
    )

    # Restore the original observation space
    #gym_wrapper.observation_space = original_obs_space

    return agent, g2op_env


def main():
    """Main evaluation script."""

    # Configuration
    checkpoint_path = "/home/adrian/Dev/NRI-for-explainable-RL-in-Power-Grids/results/experiments/test_minimal_run/CustomPPO_TEsTING_aab8cc62_2026-01-07_16-31-35"
    policy_name = "reinforcement_learning_policy"
    checkpoint_name = "checkpoint_000000"
    env_name = "l2rpn_case14_sandbox_val"
    num_episodes = 50  # Number of evaluation episodes

    # Environment configuration matching the training setup from params.json
    # todo load this from params.json automatically
    env_config = {
        "env_name": env_name,  # Will be overridden by load_rllib_agent
        "action_space": "medha",  # From params.json
        "mask": 5,
        "lib_dir": ".",  # Current directory
        "grid2op_kwargs": {
            # Will be filled by make_g2op_env
        },
        "seed": SEED,
        "rho_threshold": 0.95,
        "n_history": 1,
        "g2op_input": ["r", "t"],
        "custom_input": ["d"],
        "observation_space": "BusConnectivityGraphObsSpace",  # CRITICAL: Must match training!
        "danger": 0.9,
        "prio": False,
        "use_ffw": False,  # Set to False for evaluation
        "reset_topo": 0.9,
        "line_reco": True,
        "line_disc": False,
        "penalty_game_over": 0,
        "reward_finish": 0,
        "curriculum_training": False,
        "rules": {
            "activation_threshold": 0.95,
            "line_reco": True,
            "line_disc": False,
            "reset_topo": 0.9,
            "simulate": True
        }
    }

    # Results path
    results_path = Path(checkpoint_path) / "evaluations"

    logger.info(f"Loading agent from: {checkpoint_path}")
    logger.info(f"Checkpoint: {checkpoint_name}")
    logger.info(f"Policy: {policy_name}")

    try:
        # Load the agent
        agent, g2op_env = load_rllib_agent(
            checkpoint_path=checkpoint_path,
            policy_name=policy_name,
            checkpoint_name=checkpoint_name,
            env_name=env_name,
            env_config=env_config
        )

        logger.info(f"Agent loaded successfully!")

        # Validate observation space restoration
        from src.common.observation_space import BusConnectivityGraphObsSpace
        obs_space = agent._rllib_agent.observation_space

        if hasattr(obs_space, 'spaces') and 'reinforcement_learning_agent' in obs_space.spaces:
            rl_obs_space = obs_space.spaces['reinforcement_learning_agent']

            if isinstance(rl_obs_space, BusConnectivityGraphObsSpace):
                logger.info(f"✓ Observation space correctly restored as BusConnectivityGraphObsSpace")
                logger.info(f"  - x_dim: {rl_obs_space.x_dim}")
                logger.info(f"  - num_nodes: {rl_obs_space.num_nodes}")
                logger.info(f"  - max_num_edges: {rl_obs_space.max_num_edges}")
                logger.info(f"  - e_dim: {rl_obs_space.e_dim}")

                # Check for normalization parameters
                if hasattr(rl_obs_space, 'normalization_min') and rl_obs_space.normalization_min is not None:
                    logger.info(f"  - Normalization parameters present: Yes")
                else:
                    logger.info(f"  - Normalization parameters present: No")
            else:
                logger.warning(f"⚠ Observation space is {type(rl_obs_space).__name__}, expected BusConnectivityGraphObsSpace")
                logger.warning(f"  Custom attributes may be missing!")

        logger.info(f"Evaluating on environment: {env_name}")
        logger.info(f"Number of episodes: {num_episodes}")

        # Evaluate the agent
        evaluate_agent(
            agent=agent,
            env=g2op_env,
            path_results=results_path,
            num_episodes=num_episodes,
            verbose=True
        )

        logger.info("Evaluation completed!")

        # Load and visualize results
        logger.info("Generating visualization...")
        metrics = get_evaluation_metrics(results_path, "RLlib Agent")
        visualize_agent_survival([metrics], show=True)

        logger.info(f"Results saved to: {results_path}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    ModelCatalog.register_custom_model("gnn_model", RLlibGNNModel)
    ModelCatalog.register_custom_model("ragnn_model", RLlibRAGNNModel)
    main()

