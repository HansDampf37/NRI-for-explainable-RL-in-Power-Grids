import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from tqdm import tqdm

from evaluate_rllib_agent import load_rllib_agent, load_config
from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
from src.rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

from src.visualization import visualize_graph, PlottingArgs, get_node_styles

logger = logging.getLogger(__name__)

class AgentSpec:
    def __init__(self, name: str, load_path: Path, checkpoint_name: str, policy_name: str = "reinforcement_learning_policy"):
        self.name = name
        self.checkpoint_name = checkpoint_name
        self.policy_name = policy_name
        self.load_path = load_path

def load_agent_from_spec(agent_spec: AgentSpec, env_name: str = "l2rpn_case14_sandbox_val") -> Tuple[BaseAgent, Environment, CustomizedGrid2OpEnvironment]:
    params = load_config(agent_spec.load_path)
    env_config = params["evaluation_config"]["env_config"]
    return load_rllib_agent(
        checkpoint_path=agent_spec.load_path,
        policy_name=agent_spec.policy_name,
        checkpoint_name=agent_spec.checkpoint_name,
        env_name=env_name,
        env_config=env_config
    )


class CrossValidateResult:
    """
    Represents the result of querying a backup model in steps where model one fails.
    Attributes:
        failing_agent (AgentSpec): The first model specification.
        backup_agent (AgentSpec): The second model specification.
        additional_timesteps (Dict[str, int]): A dictionary mapping chronic ids to the number of timesteps that the backup model managed to survive after the initial model failed.
        failing_agent_completed (Dict[str, bool]): A dictionary mapping chronic ids to whether the failing agent completed the episode.
        backup_agent_completed (Dict[str, Optional[bool]]): A dictionary mapping chronic ids to whether the backup agent completed the episode (None if the failing agent was not used).
        connected_lines_before_failure (Dict[str, List[int]]): A dictionary mapping chronic ids to the list of disconnected lines before failure.
        rhos_before_failure (Dict[str, List[int]]): A dictionary mapping chronic ids to the rho values per line.
    """

    def __init__(self, failing_agent: AgentSpec, backup_agent: AgentSpec, additional_timesteps: Dict[str, Optional[int]] = None, failing_agent_completed: Dict[str, bool] = None, backup_agent_completed: Dict[str, Optional[bool]] = None):
        self.failing_agent = failing_agent
        self.backup_agent = backup_agent
        self.additional_timesteps = additional_timesteps or {}
        self.failing_agent_completed = failing_agent_completed or {}
        self.backup_agent_completed = backup_agent_completed or {}
        self.connected_lines_before_failure: Dict[str, List[int]] = {}
        self.rhos_before_failure: Dict[str, List[int]] = {}


def run_until_failure(agent: BaseAgent, g2op_env: Environment, backup_env: Environment, result: CrossValidateResult) -> Tuple[bool, str]:
    """
    Runs the specified agent on the next chronic until it fails.
    The backup environment is stepped alongside to keep them in sync. After this method the backup environment is at
    the timestep where the main environment failed.
    :param agent: The agent
    :param g2op_env: The environment
    :param backup_env: The backup environment to step alongside the main environment
    :param result: The cross-validation result object to document the results
    :return: whether the agent was able to complete the episode
    """
    obs = g2op_env.reset()
    backup_env.reset()

    chronic_id = g2op_env.chronics_handler.get_name()
    max_chronic_iter = g2op_env.max_episode_duration()

    done = False
    reward = 0
    num_steps = 0
    while True:
        action = agent.act(obs, reward=reward, done=done)
        new_obs, reward, done, info = g2op_env.step(action)
        num_steps += 1
        if done:
            if num_steps == max_chronic_iter:
                result.failing_agent_completed[chronic_id] = True
                result.connected_lines_before_failure[chronic_id] = None
                result.rhos_before_failure[chronic_id] = None
                result.additional_timesteps[chronic_id] = None
                result.backup_agent_completed[chronic_id] = None
                return True
            else:
                result.failing_agent_completed[chronic_id] = False
                result.connected_lines_before_failure[chronic_id] = obs.line_status.tolist()
                result.rhos_before_failure[chronic_id] = obs.rho.tolist()
                return False

        obs = new_obs
        # Step the backup environment as well to keep them in sync
        backup_env.step(action)


def continue_env_with_backup_agent(backup_g2op_env: Environment, backup_agent: BaseAgent, result: CrossValidateResult):
    """
    Given a backup environment that is already one step before a failing state, continue it with the backup agent until done.
    :param backup_g2op_env: the backup environment
    :param backup_agent: the backup agent
    :param result: a result object to document the results
    """
    chronic_id = backup_g2op_env.chronics_handler.get_name()
    max_chronic_iter = backup_g2op_env.max_episode_duration()

    obs = backup_g2op_env.current_obs
    reward = 0
    done = False
    num_steps = 0
    while True:
        action = backup_agent.act(obs, reward=reward, done=done)
        obs, reward, done, info = backup_g2op_env.step(action)
        num_steps += 1
        if done:
            result.additional_timesteps[chronic_id] = num_steps
            if backup_g2op_env.nb_time_step == max_chronic_iter:
                result.backup_agent_completed[chronic_id] = True
            else:
                result.backup_agent_completed[chronic_id] = False

            return


def cross_validate(failing_agent_spec: AgentSpec, backup_agent_spec: AgentSpec) -> CrossValidateResult:
    """
    Cross-validate two models by evaluating model2 on the timesteps where model1 fails.
    :param failing_agent_spec: the first model specification
    :param backup_agent_spec: the second model specification
    :return: a cross-validation result object
    """
    # load models and envs
    failing_agent_load, backup_agent_load = [load_agent_from_spec(agent_spec) for agent_spec in [failing_agent_spec, backup_agent_spec]]
    failing_agent, g2op_env, _ = failing_agent_load
    backup_agent, backup_env, _ = backup_agent_load

    # run on evaluation chronics
    result = CrossValidateResult(
        failing_agent=failing_agent_spec,
        backup_agent=backup_agent_spec,
    )
    num_validation_chronics = 50
    for _ in tqdm(range(num_validation_chronics), desc=f"Cross-validating {failing_agent_spec.name} with {backup_agent_spec.name}", unit="chronic"):
        completed = run_until_failure(failing_agent, g2op_env, backup_env, result)
        if not completed:
            # continue env from failing state with backup agent
            continue_env_with_backup_agent(backup_env, backup_agent, result)

    return result


def save_cross_validate_results(results: List[CrossValidateResult], save_path: Path):
    """
    Save the cross-validation results to a specified path as a JSON file.
    :param results: list of cross-validation results
    :param save_path: path to save the results
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        import json
        json_results = [
            {
                "failing_agent": result.failing_agent.name,
                "backup_agent": result.backup_agent.name,
                "additional_timesteps": result.additional_timesteps,
                "failing_agent_completed": result.failing_agent_completed,
                "backup_agent_completed": result.backup_agent_completed,
                "lines_before_failure": result.connected_lines_before_failure,
                "rhos_before_failure": result.rhos_before_failure,
            }
            for result in results
        ]
        json.dump(json_results, f, indent=4)


# ---------------------------------------------------------------------------
# Data-computation helpers
# ---------------------------------------------------------------------------

def compute_cross_validation_data(results: List[CrossValidateResult], save_dir: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute the cross-validation heatmap data from *results* and persist it.

    Saved files
    -----------
    cv_map.npy          – 2-D float array (failing_agents × backup_agents), NaN where no pair exists
    cv_agent_labels.npy – structured array with fields ``failing`` and ``backup`` (agent name lists)

    :param results: list of cross-validation results
    :param save_dir: directory in which to save the .npy files
    :return: ``(cv_map, all_failing_agents, all_backup_agents)``
    """
    all_failing_agents = sorted({r.failing_agent.name for r in results})
    all_backup_agents = sorted({r.backup_agent.name for r in results})

    cv_map = np.full((len(all_failing_agents), len(all_backup_agents)), np.nan, dtype=float)
    for result in results:
        i = all_failing_agents.index(result.failing_agent.name)
        j = all_backup_agents.index(result.backup_agent.name)
        valid = [s for s in result.additional_timesteps.values() if s is not None]
        cv_map[i, j] = np.mean(valid) if valid else np.nan

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "cv_map.npy", cv_map)
    np.save(save_dir / "cv_failing_agents.npy", np.array(all_failing_agents))
    np.save(save_dir / "cv_backup_agents.npy", np.array(all_backup_agents))
    logger.info("Saved cross-validation data to %s", save_dir)

    return cv_map, all_failing_agents, all_backup_agents


def compute_failing_edges_data(results: List[CrossValidateResult], save_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute per-agent mean connection rates and rho values from *results* and persist them.

    Saved files (one pair per agent, named after the agent)
    --------------------------------------------------------
    failing_edges_agent_names.npy          – 1-D array of agent names (ordering)
    failing_edges_connection_<agent>.npy   – mean connection rate per powerline  (shape: n_line)
    failing_edges_rho_<agent>.npy          – mean rho per powerline              (shape: n_line)
    failing_edges_pl_edge_index.npy        – powerline edge-index array          (shape: 2 × n_edges)
    failing_edges_powerline_edge_indices.npy – mapping powerline → edge index    (shape: n_line)

    :param results: list of cross-validation results
    :param save_dir: directory in which to save the .npy files
    :return: ``(lines_connected_before, rhos_before_failure, pl_edge_index, powerline_edge_indices)``
    """
    import grid2op

    results_with_failures = [r for r in results if r.connected_lines_before_failure]
    if not results_with_failures:
        logger.warning("No failures found – cannot compute failing-edge data.")
        return {}, {}, np.array([]), np.array([])

    lines_connected_before: Dict[str, np.ndarray] = {
        result.failing_agent.name: np.array(
            [v for v in result.connected_lines_before_failure.values() if v is not None]
        ).mean(axis=0)
        for result in results_with_failures
    }
    rhos_before_failure: Dict[str, np.ndarray] = {
        result.failing_agent.name: np.array(
            [v for v in result.rhos_before_failure.values() if v is not None]
        ).mean(axis=0)
        for result in results_with_failures
    }

    env = grid2op.make("l2rpn_case14_sandbox_val")
    obs_space = BusConnectivityGraphObsSpace(env.observation_space)
    obs = env.reset()
    gym_obs = obs_space.to_gym(obs)
    pl_edge_index = gym_obs[EDGE_INDEX]
    edge_mask = gym_obs['edge_mask']

    num_powerlines = env.n_line
    num_edges = int(edge_mask.sum())

    powerline_edge_indices: List[int] = []
    for pl_idx in range(num_powerlines):
        line_or_node = pl_idx
        line_ex_node = pl_idx + num_powerlines
        for edge_idx in range(num_edges):
            src, dst = pl_edge_index[:, edge_idx]
            if (src == line_or_node and dst == line_ex_node) or (src == line_ex_node and dst == line_or_node):
                powerline_edge_indices.append(edge_idx)
                break

    powerline_edge_indices_arr = np.array(powerline_edge_indices)

    save_dir.mkdir(parents=True, exist_ok=True)
    agent_names = list(lines_connected_before.keys())
    np.save(save_dir / "failing_edges_agent_names.npy", np.array(agent_names))
    np.save(save_dir / "failing_edges_pl_edge_index.npy", pl_edge_index)
    np.save(save_dir / "failing_edges_powerline_edge_indices.npy", powerline_edge_indices_arr)
    for agent_name in agent_names:
        safe = agent_name.replace(" ", "_")
        np.save(save_dir / f"failing_edges_connection_{safe}.npy", lines_connected_before[agent_name])
        np.save(save_dir / f"failing_edges_rho_{safe}.npy", rhos_before_failure[agent_name])
    logger.info("Saved failing-edges data to %s", save_dir)

    return lines_connected_before, rhos_before_failure, pl_edge_index, powerline_edge_indices_arr


# ---------------------------------------------------------------------------
# Pure-paint helpers (accept pre-computed arrays, no agents/envs needed)
# ---------------------------------------------------------------------------

def paint_cross_validation_results(
    cv_map: np.ndarray,
    all_failing_agents: List[str],
    all_backup_agents: List[str],
    save_path: Path,
    show: bool = False,
):
    """
    Render the cross-validation heatmap from pre-computed data and save it.

    :param cv_map: 2-D float array (failing_agents × backup_agents)
    :param all_failing_agents: ordered list of failing-agent names (row labels)
    :param all_backup_agents: ordered list of backup-agent names (column labels)
    :param save_path: where to save the figure
    :param show: whether to display the figure interactively
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cv_map, cmap="viridis")

    ax.set_xticks(np.arange(len(all_backup_agents)))
    ax.set_yticks(np.arange(len(all_failing_agents)))
    ax.set_xticklabels(all_backup_agents)
    ax.set_yticklabels(all_failing_agents)
    ax.set_xlabel("Backup Model")
    ax.set_ylabel("Failing Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(cv_map.shape[0]):
        for j in range(cv_map.shape[1]):
            if not np.isnan(cv_map[i, j]):
                ax.text(
                    j, i, f"{cv_map[i, j]:.1f}",
                    ha="center", va="center",
                    color="white" if cv_map[i, j] < np.nanmean(cv_map) else "black",
                )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg. Additional Timesteps")
    ax.set_title("Cross-Validation Results")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def paint_failing_edges_connectivity(
    lines_connected_before: Dict[str, np.ndarray],
    pl_edge_index: np.ndarray,
    powerline_edge_indices: np.ndarray,
    save_path: Path,
    show: bool = False,
):
    """
    Render one figure showing the **line connection rate** before failure for each agent.

    :param lines_connected_before: mapping agent_name → mean connection-rate array (shape: n_line)
    :param pl_edge_index: powerline edge-index array (shape: 2 × n_edges)
    :param powerline_edge_indices: mapping powerline index → edge index in pl_edge_index
    :param save_path: where to save the figure
    :param show: whether to display the figure interactively
    """
    import grid2op
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    env = grid2op.make("l2rpn_case14_sandbox_val")
    num_edges = pl_edge_index.shape[1]
    connection_cmap = plt.colormaps['RdYlGn']
    neutral_gray = '#808080'

    num_agents = len(lines_connected_before)
    fig, axes = plt.subplots(1, num_agents, figsize=(12 * num_agents, 8), constrained_layout=True)
    if num_agents == 1:
        axes = np.array([axes])

    for idx, (agent_name, connection_rates) in enumerate(lines_connected_before.items()):
        edge_colors = [neutral_gray] * num_edges
        edge_widths = [1.0] * num_edges

        for pl_idx, edge_idx in enumerate(powerline_edge_indices):
            connection_rate = connection_rates[pl_idx]
            color = connection_cmap((connection_rate - 0.9) / 0.1)
            edge_colors[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
            edge_widths[edge_idx] = 3.0

        plotting_args = PlottingArgs(
            num_nodes=57,
            node_styles=get_node_styles(env, BusConnectivityGraphObsSpace),
            powerline_edge_index=pl_edge_index,
            powerline_edge_colors=edge_colors,
            powerline_edge_widths=edge_widths,
        )
        visualize_graph(plotting_args, ax=axes[idx])
        axes[idx].set_title(f"{agent_name}\nLine Connection Before Failure")

    norm = Normalize(vmin=0.9, vmax=1.0)
    sm = ScalarMappable(cmap=connection_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical', pad=0.02, aspect=30, fraction=0.02)
    cbar.set_label('Average Connection Rate Before Failure', fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def paint_failing_edges_rho(
    rhos_before_failure: Dict[str, np.ndarray],
    pl_edge_index: np.ndarray,
    powerline_edge_indices: np.ndarray,
    save_path: Path,
    show: bool = False,
):
    """
    Render one figure showing the **line congestion (rho)** before failure for each agent.

    :param rhos_before_failure: mapping agent_name → mean rho array (shape: n_line)
    :param pl_edge_index: powerline edge-index array (shape: 2 × n_edges)
    :param powerline_edge_indices: mapping powerline index → edge index in pl_edge_index
    :param save_path: where to save the figure
    :param show: whether to display the figure interactively
    """
    import grid2op
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    env = grid2op.make("l2rpn_case14_sandbox_val")
    num_edges = pl_edge_index.shape[1]
    rho_cmap = plt.colormaps['RdYlGn']
    neutral_gray = '#808080'

    num_agents = len(rhos_before_failure)
    fig, axes = plt.subplots(1, num_agents, figsize=(12 * num_agents, 8), constrained_layout=True)
    if num_agents == 1:
        axes = np.array([axes])

    for idx, (agent_name, rhos) in enumerate(rhos_before_failure.items()):
        edge_colors = [neutral_gray] * num_edges
        edge_widths = [1.0] * num_edges

        for pl_idx, edge_idx in enumerate(powerline_edge_indices):
            rho = rhos[pl_idx]
            rho_normalized = min(1.0, max(0.0, rho))
            color = rho_cmap(1.0 - rho_normalized)
            edge_colors[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
            edge_widths[edge_idx] = 3.0

        plotting_args = PlottingArgs(
            num_nodes=57,
            node_styles=get_node_styles(env, BusConnectivityGraphObsSpace),
            powerline_edge_index=pl_edge_index,
            powerline_edge_colors=edge_colors,
            powerline_edge_widths=edge_widths,
        )
        visualize_graph(plotting_args, ax=axes[idx])
        axes[idx].set_title(f"{agent_name}\nLine Congestion Before Failure")

    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=rho_cmap.reversed(), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical', pad=0.02, aspect=30, fraction=0.02)
    cbar.set_label('Average Line Congestion (ρ) Before Failure\n(0 = Low Load, 1 = High Load/Overload)', fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level wrappers (kept for convenience / backward-compatibility)
# ---------------------------------------------------------------------------

def visualize_cross_validation_results(results: List[CrossValidateResult], save_path: Path, show: bool = False):
    """Compute cross-validation data and paint the heatmap in one step."""
    data_dir = save_path.parent
    cv_map, failing_agents, backup_agents = compute_cross_validation_data(results, data_dir)
    paint_cross_validation_results(cv_map, failing_agents, backup_agents, save_path, show=show)


def visualize_failing_edges(
    results: List[CrossValidateResult],
    save_path_connectivity: Path,
    save_path_rho: Path,
    show: bool = False,
):
    """Compute failing-edge data and paint both figures in one step."""
    data_dir = save_path_connectivity.parent
    lines_connected_before, rhos_before_failure, pl_edge_index, powerline_edge_indices = (
        compute_failing_edges_data(results, data_dir)
    )
    if not lines_connected_before:
        return
    paint_failing_edges_connectivity(lines_connected_before, pl_edge_index, powerline_edge_indices, save_path_connectivity, show=show)
    paint_failing_edges_rho(rhos_before_failure, pl_edge_index, powerline_edge_indices, save_path_rho, show=show)


# ---------------------------------------------------------------------------
# Repaint helpers – load saved .npy files and recreate figures
# ---------------------------------------------------------------------------

def repaint_cross_validation_results(data_dir: Path, save_path: Path, show: bool = False):
    """
    Load pre-computed cross-validation data from *data_dir* and repaint the heatmap.

    :param data_dir: directory containing ``cv_map.npy``, ``cv_failing_agents.npy``, ``cv_backup_agents.npy``
    :param save_path: where to save the figure
    :param show: whether to display the figure interactively
    """
    cv_map = np.load(data_dir / "cv_map.npy")
    all_failing_agents = np.load(data_dir / "cv_failing_agents.npy", allow_pickle=True).tolist()
    all_backup_agents = np.load(data_dir / "cv_backup_agents.npy", allow_pickle=True).tolist()
    paint_cross_validation_results(cv_map, all_failing_agents, all_backup_agents, save_path, show=show)


def repaint_failing_edges(
    data_dir: Path,
    save_path_connectivity: Path,
    save_path_rho: Path,
    show: bool = False,
):
    """
    Load pre-computed failing-edge data from *data_dir* and repaint both figures.

    :param data_dir: directory containing the ``failing_edges_*.npy`` files
    :param save_path_connectivity: where to save the connectivity figure
    :param save_path_rho: where to save the rho/congestion figure
    :param show: whether to display the figures interactively
    """
    agent_names: List[str] = np.load(data_dir / "failing_edges_agent_names.npy", allow_pickle=True).tolist()
    pl_edge_index = np.load(data_dir / "failing_edges_pl_edge_index.npy")
    powerline_edge_indices = np.load(data_dir / "failing_edges_powerline_edge_indices.npy")

    lines_connected_before: Dict[str, np.ndarray] = {}
    rhos_before_failure: Dict[str, np.ndarray] = {}
    for agent_name in agent_names:
        safe = agent_name.replace(" ", "_")
        lines_connected_before[agent_name] = np.load(data_dir / f"failing_edges_connection_{safe}.npy")
        rhos_before_failure[agent_name] = np.load(data_dir / f"failing_edges_rho_{safe}.npy")

    paint_failing_edges_connectivity(lines_connected_before, pl_edge_index, powerline_edge_indices, save_path_connectivity, show=show)
    paint_failing_edges_rho(rhos_before_failure, pl_edge_index, powerline_edge_indices, save_path_rho, show=show)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Cross-validate multiple models against each other and save the results.

    Step 1 – gather data:
        Runs the agents, computes summary statistics, and saves everything to
        JSON + .npy files under *results_dir*.

    Step 2 – paint figures:
        Loads the saved .npy files and creates the figures.  You can re-run
        this step independently via the ``repaint_*`` helpers.
    """
    model1 = AgentSpec(name="RAPPO", load_path="/home/adrian/Schreibtisch/1901/1901_rappo_with_anneal_different_betas/CustomPPO_0_426b7_2026-01-19_10-28-48/", checkpoint_name="checkpoint_000020")
    model2 = AgentSpec(name="MLP", load_path="/home/adrian/Schreibtisch/1901/1901_rainbow_baselines/CustomPPO_0_98414_2026-01-19_18-23-39_MLP/", checkpoint_name="checkpoint_000019")
    model3 = AgentSpec(name="GNN", load_path="/home/adrian/Schreibtisch/1901/1901_baselines/CustomPPO_0_4cbd2_2026-01-19_14-39-38_GNN/", checkpoint_name="checkpoint_000023")

    results_dir = Path("results/experiments/2601_compute_metrics")
    save_results_to = results_dir / "cross_validate_models.json"
    save_heatmap_to = results_dir / "cross_validate_models.svg"
    save_connectivity_to = results_dir / "failing_edges_connectivity.svg"
    save_rho_to = results_dir / "failing_edges_rho.svg"

    models = [model1, model2, model3]  # model4 excluded for now
    pairs = [(m1, m2) for m1, m2 in product(models, models) if m1.name != m2.name]

    # ------------------------------------------------------------------
    # Step 1: gather data
    # ------------------------------------------------------------------
    results: List[CrossValidateResult] = []
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(cross_validate, m1, m2) for (m1, m2) in pairs]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(f"Cross-validation between {result.failing_agent.name} and {result.backup_agent.name}: {result.additional_timesteps}")

    save_cross_validate_results(results, save_path=save_results_to)
    compute_cross_validation_data(results, save_dir=results_dir)
    compute_failing_edges_data(results, save_dir=results_dir)

    # ------------------------------------------------------------------
    # Step 2: paint figures (can be re-run independently via repaint_*)
    # ------------------------------------------------------------------
    repaint_cross_validation_results(results_dir, save_path=save_heatmap_to)
    repaint_failing_edges(results_dir, save_path_connectivity=save_connectivity_to, save_path_rho=save_rho_to)


if __name__ == "__main__":
    import logging
    logging.getLogger("src.ra_agents.RAFeatureExtractor").setLevel(logging.ERROR)
    logging.getLogger("pandapower.convert_format").setLevel(logging.WARNING)
    main()
