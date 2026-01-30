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

def load_agent_from_spec(agent_spec: AgentSpec) -> Tuple[BaseAgent, Environment, CustomizedGrid2OpEnvironment]:
    params = load_config(agent_spec.load_path)
    env_config = params["evaluation_config"]["env_config"]
    return load_rllib_agent(
        checkpoint_path=agent_spec.load_path,
        policy_name=agent_spec.policy_name,
        checkpoint_name=agent_spec.checkpoint_name,
        env_name="l2rpn_case14_sandbox_val",
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


def visualize_cross_validation_results(results: List[CrossValidateResult], save_path: Path, show: bool = False):
    """
    Visualize the cross-validation results as a heatmap.
    :param results: the cross-validation results
    :param save_path: where to save the figure
    :param show: whether to display the figure
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Stable ordering
    all_failing_agents = sorted(
        {r.failing_agent.name for r in results}
    )
    all_backup_agents = sorted(
        {r.backup_agent.name for r in results}
    )

    cv_map = np.full(
        (len(all_failing_agents), len(all_backup_agents)),
        np.nan,
        dtype=float,
    )

    for result in results:
        i = all_failing_agents.index(result.failing_agent.name)
        j = all_backup_agents.index(result.backup_agent.name)
        cv_map[i, j] = np.mean(list([additional_steps for additional_steps in result.additional_timesteps.values() if additional_steps is not None]))

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(cv_map, cmap="viridis")

    # Axis labels
    ax.set_xticks(np.arange(len(all_backup_agents)))
    ax.set_yticks(np.arange(len(all_failing_agents)))
    ax.set_xticklabels(all_backup_agents)
    ax.set_yticklabels(all_failing_agents)

    ax.set_xlabel("Backup Model")
    ax.set_ylabel("Failing Model")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    for i in range(cv_map.shape[0]):
        for j in range(cv_map.shape[1]):
            if not np.isnan(cv_map[i, j]):
                ax.text(
                    j, i,
                    f"{cv_map[i, j]:.1f}",
                    ha="center", va="center",
                    color="white" if cv_map[i, j] < np.nanmean(cv_map) else "black"
                )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg. Additional Timesteps")

    ax.set_title("Cross-Validation Results")

    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def visualize_failing_edges(results: List[CrossValidateResult], save_path: Path, show: bool = False):
    """
    Visualize the failing and overloaded edges (powerlines) from the cross-validation results.

    The obvious hypothesis is that powerlines that are disconnected after failure are the cause of the failure.

    :param results: the cross-validation results
    :param save_path: the path to save the image
    :param show: whether to display the image
    """
    # Filter out results where agents completed successfully (no failures to analyze)
    results_with_failures = [r for r in results if r.connected_lines_before_failure]

    if not results_with_failures:
        logger.warning("No failures found in cross-validation results. Cannot visualize failing edges.")
        return

    lines_connected_before = {
        result.failing_agent.name: np.array([
            vals for vals in result.connected_lines_before_failure.values() if vals is not None]
        ).mean(axis=0)
        for result in results_with_failures
    }
    rhos_before_failure = {
        result.failing_agent.name: np.array([
            vals for vals in result.rhos_before_failure.values() if vals is not None
        ]).mean(axis=0)
        for result in results_with_failures
    }

    import grid2op
    import matplotlib.pyplot as plt

    env = grid2op.make("l2rpn_case14_sandbox_val")
    obs_space = BusConnectivityGraphObsSpace(env.observation_space)
    obs = env.reset()
    gym_obs = obs_space.to_gym(obs)
    pl_edge_index = gym_obs[EDGE_INDEX]
    edge_mask = gym_obs['edge_mask']

    # Number of unique powerlines (undirected edges)
    num_powerlines = env.n_line

    # Count actual edges (respecting edge_mask)
    num_edges = edge_mask.sum()

    # Build mapping from powerline index to edge indices in pl_edge_index
    # Powerlines connect node i (line_or) to node i+n_line (line_ex)
    # We need to find which edges in pl_edge_index correspond to powerlines
    powerline_edge_indices = []
    for pl_idx in range(num_powerlines):
        line_or_node = pl_idx
        line_ex_node = pl_idx + num_powerlines
        # Find edges connecting these nodes (bidirectional)
        for edge_idx in range(num_edges):
            src, dst = pl_edge_index[:, edge_idx]
            if (src == line_or_node and dst == line_ex_node) or (src == line_ex_node and dst == line_or_node):
                powerline_edge_indices.append(edge_idx)
                break  # Found the edge for this powerline

    # Create 2-row subplots for each agent (row 1: connections, row 2: rho)
    num_agents = len(lines_connected_before)
    fig, axes = plt.subplots(2, num_agents, figsize=(12 * num_agents, 16), constrained_layout=True)

    # Handle case of single agent (ensure axes is 2D)
    if num_agents == 1:
        axes = axes.reshape(2, 1)

    connection_cmap = plt.colormaps['RdYlGn']  # Red (disconnected) -> Yellow -> Green (connected)
    rho_cmap = plt.colormaps['RdYlGn']  # Red (high load) -> Yellow -> Green (low load)
    neutral_gray = '#808080'  # Gray for non-powerline edges

    for idx, (agent_name, connection_rates) in enumerate(lines_connected_before.items()):
        rhos = rhos_before_failure[agent_name]

        # --- First row: Connection status by color ---
        # Create color array for ALL edges, default to gray
        edge_colors_connection = [neutral_gray] * num_edges
        # Create width array for ALL edges, default to thin
        edge_widths_connection = [1.0] * num_edges

        # Set colors and widths for powerline edges based on connection data
        for pl_idx, edge_idx in enumerate(powerline_edge_indices):
            connection_rate = connection_rates[pl_idx]
            # connection_rate: 1.0 = always connected (green), 0.0 = always disconnected (red)
            color = connection_cmap(connection_rate)
            edge_colors_connection[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
            edge_widths_connection[edge_idx] = 3.0  # Thicker for powerlines

        plotting_args_connection = PlottingArgs(
            num_nodes=57,
            node_styles=get_node_styles(env, BusConnectivityGraphObsSpace),
            powerline_edge_index=pl_edge_index,
            powerline_edge_colors=edge_colors_connection,
            powerline_edge_widths=edge_widths_connection,
        )

        visualize_graph(plotting_args_connection, ax=axes[0, idx])
        axes[0, idx].set_title(f"{agent_name}\nLine Connection Before Failure")

        # --- Second row: Rho (load) by color ---
        # Create color array for ALL edges, default to gray
        edge_colors_rho = [neutral_gray] * num_edges
        # Create width array for ALL edges, default to thin
        edge_widths_rho = [1.0] * num_edges

        # Set colors and widths for powerline edges based on rho data
        for pl_idx, edge_idx in enumerate(powerline_edge_indices):
            rho = rhos[pl_idx]
            # Normalize rho to [0, 1] for colormap
            # rho: 0.0 = no load (green), 1.0 = at limit (red)
            rho_normalized = min(1.0, max(0.0, rho))  # Clamp to [0, 1]
            color = rho_cmap(1.0 - rho_normalized)  # Invert: high rho = red (low in colormap)
            edge_colors_rho[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
            edge_widths_rho[edge_idx] = 3.0  # Thicker for powerlines

        plotting_args_rho = PlottingArgs(
            num_nodes=57,
            node_styles=get_node_styles(env, BusConnectivityGraphObsSpace),
            powerline_edge_index=pl_edge_index,
            powerline_edge_colors=edge_colors_rho,
            powerline_edge_widths=edge_widths_rho,
        )

        visualize_graph(plotting_args_rho, ax=axes[1, idx])
        axes[1, idx].set_title(f"{agent_name}\nLine Congestion Before Failure")

    # Add colorbars for each row (only once, outside the loop)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Colorbar for connection status (row 0) - vertical on the right
    norm_connection = Normalize(vmin=0, vmax=1)
    sm_connection = ScalarMappable(cmap=connection_cmap, norm=norm_connection)
    sm_connection.set_array([])
    cbar_connection = fig.colorbar(sm_connection, ax=axes[0, :].tolist(), orientation='vertical',
                                    pad=0.15, aspect=20, fraction=0.02)
    cbar_connection.set_label('Average Connection Rate Before Failure\n(0 = Always Disconnected, 1 = Always Connected)', fontsize=10)

    # Colorbar for rho (row 1) - vertical on the right
    # We use reversed colormap since we map high rho (1.0) -> red by using (1.0 - rho) with RdYlGn
    norm_rho = Normalize(vmin=0, vmax=1)
    sm_rho = ScalarMappable(cmap=rho_cmap.reversed(), norm=norm_rho)
    sm_rho.set_array([])
    cbar_rho = fig.colorbar(sm_rho, ax=axes[1, :].tolist(), orientation='vertical',
                            pad=0.15, aspect=20, fraction=0.02)
    cbar_rho.set_label('Average Line Congestion (ρ) Before Failure\n(0 = Low Load, 1 = High Load/Overload)', fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)


def main():
    """
    Cross-validate multiple models against each other and save the results.
    1. Define model specifications (RAPPO, MLP, GNN, NRIGNN).
    2. For each pair of models, perform cross-validation if they are different.
    3. Save the results to a JSON file.
    """
    model1 = AgentSpec(name="RAPPO", load_path="/home/adrian/Schreibtisch/1901/1901_rappo_with_anneal_different_betas/CustomPPO_0_426b7_2026-01-19_10-28-48/", checkpoint_name="checkpoint_000020")
    model2 = AgentSpec(name="MLP", load_path="/home/adrian/Schreibtisch/1901/1901_rainbow_baselines/CustomPPO_0_98414_2026-01-19_18-23-39_MLP/", checkpoint_name="checkpoint_000019")
    model3 = AgentSpec(name="GNN", load_path="/home/adrian/Schreibtisch/1901/1901_baselines/CustomPPO_0_4cbd2_2026-01-19_14-39-38_GNN/", checkpoint_name="checkpoint_000023")
    model4 = AgentSpec(name="NRIGNN", load_path="/home/adrian/Schreibtisch/1901/1901_baselines/CustomPPO_0_4eebd_2026-01-19_14-39-41_NRI/", checkpoint_name="checkpoint_000024")

    save_results_to = Path("results/experiments/2601_compute_metrics/cross_validate_models.json")
    save_figure_to = Path("results/experiments/2601_compute_metrics/cross_validate_models.svg")
    save_figure2_to = Path("results/experiments/2601_compute_metrics/failing_edges.svg")

    models = [model1, model2, model3] #, model2, model3, model4]
    pairs = [(m1, m2) for m1, m2 in product(models, models) if m1.name != m2.name]

    results: List[CrossValidateResult] = []
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(cross_validate, m1, m2) for (m1, m2) in pairs]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(f"Cross-validation between {result.failing_agent.name} and {result.backup_agent.name}: {result.additional_timesteps}")

    save_cross_validate_results(results, save_path=save_results_to)
    visualize_cross_validation_results(results, save_path=save_figure_to)
    visualize_failing_edges(results, save_path=save_figure2_to)


if __name__ == "__main__":
    import logging
    logging.getLogger("src.ra_agents.RAFeatureExtractor").setLevel(logging.ERROR)
    logging.getLogger("pandapower.convert_format").setLevel(logging.WARNING)
    main()
