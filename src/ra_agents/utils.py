from pathlib import Path
from typing import Optional, Callable, Dict, Any

from grid2op.gym_compat import BoxGymObsSpace
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from src.common.baseline_agent import BaselineAgent, evaluate_agent, evaluate_sb3_alg, TopologyPolicy
from src.common.constants import logger
from src.common.env import G2OpGymEnv
from src.visualization import get_evaluation_metrics, visualize_agent_survival


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


def evaluate(algorithm: BaseAlgorithm, topology_policy: TopologyPolicy, env_creation, path_results: Path, cfg: DictConfig) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Evaluates an algorithm and associated agent on train/test/val envs.
    Stores results and plots in path_results, and returns a dict of metrics:
    {
      "agent": {
        "train"|"test"|"val": {"survival_duration": List[int], "returns": List[float]}
      },
      "rl_algorithm": {
        "train"|"test"|"val": {"survival_duration": List[int], "returns": List[float]}
      }
    }
    """
    # Run evaluations and persist artifacts
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

    # Collect metrics
    metrics_agent = [get_evaluation_metrics(Path(path_results, "agent", dataset), dataset) for dataset in ["train", "test", "val"]]
    metrics_topo_policy = [get_evaluation_metrics(Path(path_results, "rl_algorithm", dataset), dataset) for dataset in ["train", "test", "val"]]

    # Generate summary plots
    # Include two heuristic baselines for agent summary
    metrics_agent_with_baselines = metrics_agent + [
        get_evaluation_metrics(Path("data/evaluations/heuristic_agents/reco_powerline_agent/train"), "Reconnect Powerline"),
        get_evaluation_metrics(Path("data/evaluations/heuristic_agents/do_nothing_agent/train"), "Do Nothing"),
    ]
    visualize_agent_survival(metrics_agent_with_baselines, Path(path_results, "agent_summary.png"), show=False)
    visualize_agent_survival(metrics_topo_policy, Path(path_results, "rl_algorithm_summary.png"), show=False)

    # Convert to a plain dict for downstream consumption (e.g., Optuna)
    datasets = ["train", "test", "val"]
    result: Dict[str, Dict[str, Dict[str, Any]]] = {"agent": {}, "rl_algorithm": {}}
    for i, ds in enumerate(datasets):
        result["agent"][ds] = {
            "survival_duration": metrics_agent[i].survival_duration,
            "returns": metrics_agent[i].returns,
        }
        result["rl_algorithm"][ds] = {
            "survival_duration": metrics_topo_policy[i].survival_duration,
            "returns": metrics_topo_policy[i].returns,
        }

    return result


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq: int, topology_policy: TopologyPolicy, env_fn: Callable[[DictConfig, str], G2OpGymEnv], path_results_root: Path, cfg: DictConfig, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.env_fn = env_fn
        self.path_results_root = path_results_root
        self.cfg = cfg
        self.topology_policy = topology_policy

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            logger.info(f"Evaluation for step {self.num_timesteps}")
            path = Path(self.path_results_root, f"checkpoint_{self.num_timesteps}")
            # Trigger evaluation; return value is ignored here as artifacts are saved to disk
            _ = evaluate(
                algorithm=self.model,
                topology_policy=self.topology_policy,
                env_creation=self.env_fn,
                path_results=path,
                cfg=self.cfg,
            )
        return True

    def _on_training_end(self) -> None:
        for setup in ["agent", "rl_algorithm"]:
            for dataset in ["train", "test", "val"]:
                # Collect all checkpoint directories and sort by step count
                if not self.path_results_root.exists():
                    logger.warning(f"Path '{self.path_results_root}' does not exist. No training summary generated.")
                    return

                checkpoints = []
                for p in Path(self.path_results_root).iterdir():
                    if p.is_dir() and p.name.startswith("checkpoint_"):
                        try:
                            step = int(p.name.split("_", 1)[1])
                            checkpoints.append((step, Path(p, setup)))
                        except ValueError:
                            logger.debug(f"Ignoring directory '{p}' (invalid step number).")

                if not checkpoints:
                    logger.warning("No checkpoints found. No training summary generated.")
                    return

                checkpoints.sort(key=lambda x: x[0])

                # Create metrics per checkpoint and visualize progression
                metrics = []
                for step, path in checkpoints:
                    metrics.append(get_evaluation_metrics(Path(path, dataset), f"step_{step}"))

                out_path = Path(self.path_results_root, f"{setup}_{dataset}_training_effect.png")
                visualize_agent_survival(metrics, out_path, show=False)
                logger.info(f"Training summary saved at '{out_path}'.")