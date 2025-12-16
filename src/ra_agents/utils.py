import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from grid2op.gym_compat import BoxGymObsSpace
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from src.common.baseline_agent import BaselineAgent, evaluate_agent, evaluate_sb3_alg
from src.common.env import G2OpGymEnv
from src.visualization import get_evaluation_metrics, visualize_agent_survival

logger = logging.getLogger(__name__)


def get_env(cfg: DictConfig, env_name: Optional[str] = None) -> G2OpGymEnv:
    """
    Creates a Grid2opWrapperEnvironment with fitting action and observation spaces from hydra config.

    :param cfg: Hydra config
    :param env_name: Optional override for the environment name
    :return: The environment
    """
    env: G2OpGymEnv = G2OpGymEnv(
        cfg.env.training_env.env_name if env_name is None else env_name,
        rule_config=cfg.env.rule_config,
        obs_space_creation=lambda e: instantiate(cfg.rl.obs_space, grid2op_observation_space=e.observation_space),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space),
        curriculum_level_settings=cfg.env.training_env.curriculum_level_settings,
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
        rule_config=cfg.env.rule_config,
        obs_space_creation=lambda e: BoxGymObsSpace(grid2op_observation_space=e.observation_space, attr_to_keep=["rho", "topo_vect"]),
        act_space_creation=lambda e: instantiate(cfg.rl.act_space, grid2op_action_space=e.action_space),
        curriculum_level_settings=cfg.env.training_env.curriculum_level_settings,
    )
    return env


class CurriculumCallback(BaseCallback):
    """
    Generic curriculum learning callback switching environment difficulty levels.

    Arguments:
    - total_timesteps: total training timesteps to compute progress thresholds
    - level2_at_fraction: switch to level 2 when progress >= this fraction (default 1/5)
    - level3_at_fraction: switch to level 3 when progress >= this fraction (default 7/15)
    - start_level: initial level set externally on env creation (default 1)

    This callback will call `set_curriculum(level)` on the underlying environment(s) via VecEnv.env_method.
    """
    def __init__(self, total_timesteps: int, level1_at_fraction: float = 1.0 / 5.0, level2_at_fraction: float = 7.0 / 15.0, start_level: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.level1_timestep = int(float(level1_at_fraction) * self.total_timesteps)
        self.level2_timestep = int(float(level2_at_fraction) * self.total_timesteps)
        self.start_level = int(start_level)
        self._switched_to_1 = False
        self._switched_to_2 = False

    def _on_training_start(self) -> None:
        # ensure env starts at requested level
        self.model.get_env().env_method("set_curriculum", int(self.start_level))

    def _on_step(self) -> bool:
        t = int(self.model.num_timesteps)
        if not self._switched_to_2 and t >= self.level2_timestep:
            self.model.get_env().env_method("set_curriculum", 2)
            self._switched_to_2 = self._switched_to_1 = True
        elif not self._switched_to_1 and t >= self.level1_timestep:
            self.model.get_env().env_method("set_curriculum", 1)
            self._switched_to_1 = True
        return True


def evaluate(algorithm: BaseAlgorithm, env_creation, path_results: Path, cfg: DictConfig, verbose = True, final: bool= True) -> Dict[str, Dict[str, Dict[str, Any]]]:
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
    :param algorithm: The algorithm to evaluate
    :param env_creation: Function that creates the environment
    :param path_results: Path where the results will be stored
    :param cfg: Hydra config
    :param verbose: print extra explanatory or diagnostic information
    :param final: if this argument is true, the evaluation goes on for longer (the config inside rl.eval.final is used)
    :return metrics: Dict of metrics
    """
    # Run evaluations and persist artifacts
    for dataset in ["train", "test", "val"]:
        env_dataset: G2OpGymEnv = env_creation(cfg, f"{cfg.env.name}_{dataset}")
        agent = BaselineAgent(
            g2op_action_space=env_dataset._g2op_env.action_space,
            rl_policy=algorithm.policy,
            rule_config=cfg.env.rule_config
        )
        evaluate_agent(
            agent=agent,
            env=env_dataset._g2op_env,
            path_results=Path(path_results, "agent", dataset),
            num_episodes=cfg.rl.eval.final.nb_episodes if final else cfg.rl.eval.during_training.nb_episodes,
            max_episode_length=cfg.rl.eval.final.max_episode_length if final else cfg.rl.eval.during_training.max_episode_length,
            verbose=verbose
        )
        evaluate_sb3_alg(
            alg=algorithm,
            env=env_dataset,
            path_results=Path(path_results, "rl_algorithm", dataset),
            num_episodes=cfg.rl.eval.final.nb_episodes if final else cfg.rl.eval.during_training.nb_episodes,
            max_episode_length=cfg.rl.eval.final.max_episode_length if final else cfg.rl.eval.during_training.max_episode_length,
            verbose=verbose
        )

    # Collect metrics
    metrics_agent = [get_evaluation_metrics(Path(path_results, "agent", dataset), dataset) for dataset in ["train", "test", "val"]]
    metrics_topo_policy = [get_evaluation_metrics(Path(path_results, "rl_algorithm", dataset), dataset) for dataset in ["train", "test", "val"]]

    # Generate summary plots
    # Include two heuristic baselines for agent summary
    metrics_agent_with_baselines = metrics_agent + [
        get_evaluation_metrics(Path("results/evaluations/heuristic_agents/reco_powerline_agent/train"), "Reconnect Powerline"),
        get_evaluation_metrics(Path("results/evaluations/heuristic_agents/do_nothing_agent/train"), "Do Nothing"),
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
    def __init__(self, eval_freq: int, env_fn: Callable[[DictConfig, str], G2OpGymEnv], path_results_root: Path, cfg: DictConfig, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.env_fn = env_fn
        self.path_results_root = path_results_root
        self.cfg = cfg

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            if self.verbose > 0:
                logger.info(f"Evaluation for step {self.num_timesteps}")

            path = Path(self.path_results_root, f"checkpoint_{self.num_timesteps}")
            # Trigger evaluation; return value is ignored here as artifacts are saved to disk
            _ = evaluate(
                algorithm=self.model,
                env_creation=self.env_fn,
                path_results=path,
                cfg=self.cfg,
                verbose=self.verbose > 0,
                final=False
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
                if self.verbose > 0:
                    logger.info(f"Training summary saved at '{out_path}'.")

def get_callbacks(cfg: DictConfig, path_results: Path, env_creation: Callable = get_env) -> List[BaseCallback]:
    callbacks = []
    if cfg.rl.eval.during_training.active:
        eval_callback = EvalCallback(
            eval_freq=max(cfg.rl.train.timesteps // cfg.rl.eval.during_training.num_evaluations_during_training, 1),
            env_fn=env_creation,
            path_results_root=Path(path_results, "checkpoints"),
            cfg=cfg,
            verbose=1 if cfg.rl.verbose else 0
        )
        callbacks.append(eval_callback)

    if cfg.rl.train.curriculum_level_config.active:
        curriculum_cb = CurriculumCallback(
            total_timesteps=cfg.rl.train.timesteps,
            level1_at_fraction=float(cfg.rl.train.curriculum_level_config.level1_at_fraction),
            level2_at_fraction=float(cfg.rl.train.curriculum_level_config.level2_at_fraction),
            start_level=int(cfg.rl.train.curriculum_level_config.start_level),
            verbose=1 if cfg.rl.verbose else 0,
        )
        callbacks.append(curriculum_cb)

    return callbacks