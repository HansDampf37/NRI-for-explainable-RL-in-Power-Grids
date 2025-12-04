from pathlib import Path
from typing import Optional

from grid2op.gym_compat import BoxGymObsSpace
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm

from common.baseline_agent import BaselineAgent, evaluate_agent, evaluate_sb3_alg, TopologyPolicy
from common.env import G2OpGymEnv
from visualization import get_evaluation_metrics, visualize_agent_survival


def get_env(cfg, env_name: Optional[str] = None) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action and observation spaces from hydra config.

    :param cfg: The hydra config
    :param env_name: Optional override for the environment name
    :return: The environment
    """
    env: G2OpGymEnv = G2OpGymEnv(
        cfg.env.training_env.env_name if env_name is None else env_name,
        safe_max_rho=cfg.env.safe_max_rho,
        obs_space_creation=lambda e: instantiate(cfg.rl.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space)
    )
    return env

def get_env_mlp_baseline(cfg, env_name: Optional[str] = None) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action and observation spaces from hydra config.

    :param cfg: The hydra config
    :param env_name: Optional override for the environment name
    :return: The environment
    """
    env: G2OpGymEnv = G2OpGymEnv(
        cfg.env.training_env.env_name if env_name is None else env_name,
        safe_max_rho=cfg.env.safe_max_rho,
        obs_space_creation=lambda e: BoxGymObsSpace(grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space)
    )
    return env


def evaluate(algorithm: BaseAlgorithm, topology_policy: TopologyPolicy, env_creation, path_results: Path, cfg: DictConfig):
    """
    Evaluates an algorithm on test train and validation envs.
    Evaluates the associated agent on test train and validation envs.
    Stores results together with plots in the path_results folder.
    @param algorithm: the algorithm to evaluate
    @param topology_policy: the topology policy (used inside the agent)
    @param env_creation: a method that takes an env cfg, name and returns a G2OpGymEnv
    @param path_results: where to store the results
    @param cfg: the hydra config
    """
    for dataset in ["train", "test", "val"]:
        env_dataset: G2OpGymEnv = env_creation(cfg, f"{cfg.env.name}_{dataset}")
        agent = BaselineAgent(
            env_dataset._g2op_env.action_space,
            topology_policy,
            k=cfg.env.agent_k,
            safe_max_rho=cfg.env.safe_max_rho,
        )
        evaluate_agent(
            agent=agent,
            env=env_dataset._g2op_env,
            path_results=Path(path_results, "agent", dataset),
            num_episodes=cfg.rl.eval.nb_episodes,
            max_episode_length=cfg.rl.eval.max_episode_length,
        )
        evaluate_sb3_alg(
            alg=algorithm,
            env=env_dataset,
            path_results=Path(path_results, "rl_algorithm", dataset),
            num_episodes=cfg.rl.eval.nb_episodes,
            max_episode_length=cfg.rl.eval.max_episode_length,
        )

    metrics_agent = [get_evaluation_metrics(Path(path_results, "agent", dataset), dataset) for dataset in ["train", "test", "val"]]
    metrics_agent += [get_evaluation_metrics(Path("data/evaluations/heuristic_agents/reco_powerline_agent/train"), "Reconnect Powerline")]
    metrics_agent += [get_evaluation_metrics(Path("data/evaluations/heuristic_agents/do_nothing_agent/train"), "Do Nothing")]
    metrics_topo_policy = [get_evaluation_metrics(Path(path_results, "rl_algorithm", dataset), dataset) for dataset in ["train", "test", "val"]]
    visualize_agent_survival(metrics_agent, Path(path_results, "agent_summary.png"), show=False)
    visualize_agent_survival(metrics_topo_policy, Path(path_results, "rl_algorithm_summary.png"), show=False)
