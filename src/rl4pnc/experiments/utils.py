"""
Utilities in the grid2op experiments.
"""

import json
import logging
import os
from datetime import datetime
from time import time
from typing import Any, Dict, List, OrderedDict, Union

import numpy as np
import pandas as pd
import ray
from grid2op.Environment import BaseEnv
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.experiment import Trial
from ray.tune.result_grid import ResultGrid
from ray.tune.stopper.stopper import Stopper
from tabulate import tabulate

from src.rl4pnc.algorithms.custom_ppo import CustomPPO
from src.rl4pnc.algorithms.optuna_search import MyOptunaSearch
from src.rl4pnc.experiments.callback import Style, TuneCallback

REPORT_END = True


def calculate_action_space_asymmetry(env: BaseEnv, add_dn: bool = False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies without symmetries.
    """

    nr_substations = len(env.sub_info)

    logging.info("no symmetries")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )

        alpha = 2 ** (nr_elements - 1) - (2 ** nr_non_lines - 1)
        action_space += alpha if alpha > 1 else 0
        # if alpha > 1:  # without do nothings for single substations
        if (add_dn and alpha > 0) or (alpha > 1):
            controllable_substations[sub] = alpha
        possible_topologies *= max(alpha, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_medha(env: BaseEnv, add_dn: bool = False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies following Subrahamian (2021).
    """
    nr_substations = len(env.sub_info)

    logging.info("medha")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )
        alpha = 2 ** (nr_elements - 1)
        beta = nr_elements - (1 if nr_elements == 2 else 0)
        gamma = 2 ** nr_non_lines - 1 - nr_non_lines
        combined = alpha - beta - gamma
        action_space += combined if combined > 1 else 0
        # if combined > 1:  # without do nothings for single substations
        if (add_dn and combined > 0) or (combined > 1):
            controllable_substations[sub] = combined
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    print(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_tennet(env: BaseEnv, add_dn=False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies following the proposed action space.
    """
    nr_substations = len(env.sub_info)

    logging.info("TenneT")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )
        nr_lines = nr_elements - nr_non_lines

        combined = (
                           (
                                   2 ** nr_non_lines - 2
                           )  # configuratations of non-lines except when all lines are same colour
                           * (
                                   2 ** nr_lines  # configurations of lines
                                   - 2 * nr_lines  # minus lines that there is exactly one line at a busbar
                                   - 2  # minus case where all lines have the same colour
                                   + (2 if nr_lines == 1 else 0)  # due to doubles with 1 line
                                   + (2 if nr_lines == 2 else 0)  # due to doubles with 2 lines
                           )
                           + 2  # configurations where non-lines all have the same colour
                           * (
                                   2 ** nr_lines  # configurations of lines
                                   - 2 * nr_lines  # minus lines that there is exactly one line at a busbar
                                   - 1  # if all non-lines have the same colour, then if all lines are also this colour, it's allowed
                                   + (2 if nr_lines == 2 else 0)  # due to doubles with 2 lines
                                   + (1 if nr_lines == 1 else 0)  # due to doubles with 1 line
                           )
                   ) / 2  # remove symmetries

        action_space += int(combined) if combined > 1 else 0
        if (add_dn and combined > 0) or (combined > 1):  # combined > 1: without do nothings for single substations
            controllable_substations[sub] = combined
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def get_capa_substation_id(
        line_info: dict[int, list[int]],
        obs_batch: Union[List[Dict[str, Any]], Dict[str, Any]],
        controllable_substations: dict[int, int],
) -> list[int]:
    """
    Returns the substation id of the substation to act on according to CAPA.
    """
    # calculate the mean rho per substation
    connected_rhos: dict[int, list[float]] = {agent: [] for agent in line_info}
    for sub_idx in line_info:
        for line_idx in line_info[sub_idx]:
            if isinstance(obs_batch, OrderedDict):
                connected_rhos[sub_idx].append(
                    obs_batch["previous_obs"]["rho"][0][line_idx]
                    # obs_batch["original_obs"]["rho"][0][line_idx]
                )
            elif isinstance(obs_batch, dict):
                connected_rhos[sub_idx].append(
                    obs_batch["previous_obs"]["rho"][line_idx]
                    # obs_batch["original_obs"]["rho"][line_idx]
                )
            else:
                raise ValueError("The observation batch is not supported.")
    for sub_idx in connected_rhos:
        connected_rhos[sub_idx] = [float(np.mean(connected_rhos[sub_idx]))]

    # set non-controllable substations to 0
    for sub_idx in connected_rhos:
        if sub_idx not in list(controllable_substations.keys()):
            connected_rhos[sub_idx] = [0.0]

    # order the substations by the mean rho, maximum first
    connected_rhos = dict(
        sorted(connected_rhos.items(), key=lambda item: item[1], reverse=True)
    )

    # # find substation with max average rho
    # max_value = max(connected_rhos.values())
    # return [key for key, value in connected_rhos.items() if value == max_value][0]

    # return the ordered entries
    # NOTE: When there are two equal max values, the first one is returned first
    return list(connected_rhos.keys())


def find_list_of_agents(env: BaseEnv, action_space: str) -> dict[int, int]:
    """
    Function that returns the number of controllable substations.
    """
    add_dn = "dn" in action_space
    if action_space.startswith("asymmetry"):
        _, _, list_of_agents = calculate_action_space_asymmetry(env, add_dn)
        return list_of_agents
    if action_space.startswith("medha"):
        _, _, list_of_agents = calculate_action_space_medha(env, add_dn)
        return list_of_agents
    if action_space.startswith("tennet"):
        _, _, list_of_agents = calculate_action_space_tennet(env, add_dn)
        return list_of_agents
    raise ValueError("The action space is not supported.")


def find_substation_per_lines(
        env: BaseEnv, list_of_agents: list[int]
) -> dict[int, list[int]]:
    """
    Returns a dictionary connecting line ids to substations.
    """
    line_info: dict[int, list[int]] = {agent: [] for agent in list_of_agents}
    for sub_idx in list_of_agents:
        for or_id in env.observation_space.get_obj_connect_to(substation_id=sub_idx)[
            "lines_or_id"
        ]:
            line_info[sub_idx].append(or_id)
        for ex_id in env.observation_space.get_obj_connect_to(substation_id=sub_idx)[
            "lines_ex_id"
        ]:
            line_info[sub_idx].append(ex_id)

    return line_info


def delete_nested_key(d, path):
    keys = path.split('/')
    current = d

    # Traverse through the dictionary using keys from the path
    for key in keys[:-1]:  # Iterate until the second last key
        if key in current:
            current = current[key]
        else:
            return  # If any key is missing, return without making changes

    # Now current points to the dictionary containing the key to be deleted
    last_key = keys[-1]
    if last_key in current:
        del current[last_key]


class MaxCustomMetricStopper(Stopper):
    """Stop trials after reaching a maximum value for the custom metric

    Args:
        metric: Metric to use.
        max_value: If custom metric reaches this value stop trials
    """

    def __init__(self, metric: str, max_value: int):
        self._max_value = max_value
        self._custom_Metric = metric

    def __call__(self, trial_id: str, result: Dict):
        print("current value custom metric: ", result["custom_metric"][self._custom_Metric])
        return result["custom_metric"][self._custom_Metric] >= self._max_value

    def stop_all(self):
        return False


class TimeStopper(Stopper):
    def __init__(self, deadline):
        self._start = time()
        if isinstance(deadline, str):
            self._deadline = int(deadline.split(":")[0]) * 3600 + int(deadline.split(":")[1]) * 60
            print("Run training for ", deadline, " hours.")
        else:
            self._deadline = deadline * 60  # Stop all trials after deadline minutes
            print("Run training for ", deadline, " minutes.")

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time() - self._start > self._deadline


def get_duration(setup):
    deadline = setup.get("duration", 0)
    # convert deadline to seconds
    if deadline == 0:
        print(f"Run until {setup['nb_timesteps']} agent time steps.")
        return deadline
    if isinstance(deadline, str):
        deadline = int(deadline.split(":")[0]) * 3600 + int(deadline.split(":")[1]) * 60
    else:
        deadline = deadline * 60  # Stop all trials after deadline minutes
    print("Run training for ", deadline, " seconds.")
    return deadline


def trial_str_creator(trial: Trial, job_id=""):
    # Don't modify trial.trial_id as it breaks Optuna's internal tracking!
    # Just create a custom display name
    base_id = trial.trial_id.split("_")[0]
    if job_id:
        custom_id = "{}_{}".format(job_id, base_id)
    else:
        custom_id = base_id
    print('Creating trial with ID: ', custom_id)
    return "{}_{}".format(trial.trainable_name, custom_id)


def trial_dir_name(trial: Trial):
    print("Trial name is: ", trial.custom_trial_name)
    return "{}_{}".format(trial.custom_trial_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


def run_training(config: dict[str, Any], setup: dict[str, Any], job_id: str) -> ResultGrid:
    """
    Function that runs the training script.
    """
    # runtime_env = {"env_vars": {"PYTHONWARNINGS": "ignore"}}
    # ray.init(runtime_env= runtime_env, local_mode=False)
    # init ray
    # Set the environment variable
    os.environ["RAY_DEDUP_LOGS"] = "0"
    # os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    # os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    # Run wandb offline and to sync when finished use following command in result directory:
    # for d in $(ls -t -d */); do cd $d; wandb sync --sync-all; cd ..; done
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    tmp_dir = ray._private.utils.get_ray_temp_dir()
    print(f"Ray's temporary directory: {tmp_dir}")
    ray.init()
    print("Ray initialization succeeded.")

    # Get the hostname and port
    address = ray.worker._real_worker._global_node.address
    host_name, port = address.split(":")
    print("Hostname:", host_name)
    print("Port:", port)

    # Use Optuna search algorithm to find good working parameters
    algo = None
    if setup['optimize']:
        points_to_eval = setup.get('points_to_evaluate', None)
        algo = MyOptunaSearch(
            metric=setup["score_metric"],
            mode="max",
            points_to_evaluate=[points_to_eval] if points_to_eval is not None else None,
        )
        if 'result_dir' in setup.keys():
            print("Retrieving results old experiment from : ", setup['result_dir'])
            algo.restore_from_dir(setup['result_dir'])
            for key in algo._space.keys():
                if '/' in key:
                    delete_nested_key(config, key)
                else:
                    del config[key]
        # # Scheduler determines if we should prematurely stop a certain experiment - NOTE: DOES NOT WORK AS EXPECTED!!!
        # scheduler = MedianStoppingRule(
        #     time_attr="timesteps_total", #Default = "time_total_s"
        #     metric=setup["score_metric"],
        #     mode="max",
        #     grace_period=setup["grace_period"], # First exploration before stopping
        #     min_samples_required=5, # Default = 3
        #     min_time_slice=10_000,
        #     hard_stop=False,
        # )
    dur = get_duration(setup)

    # Get time budget for entire optimization (different from per-trial duration)
    time_budget = setup.get("time_budget_s", None)
    if time_budget is None and setup.get("optimize", False):
        # For optimization, calculate time budget from duration if specified
        if dur:
            # Leave some buffer time (10%) for cleanup before SLURM kills the job
            time_budget = int(dur * 0.9)
            print(f"Setting time budget for optimization to {time_budget}s ({time_budget/3600:.2f} hours) with 10% buffer for cleanup")

    storage_path = os.path.abspath(os.path.join(setup.get("workdir", "."), "results", "experiments"))
    os.makedirs(storage_path, exist_ok=True)
    print(f"Results will be saved to: {storage_path}/{setup['experiment_name']}")

    # Create tuner
    tuner = tune.Tuner(
        trainable=CustomPPO,
        param_space=config,
        run_config=air.RunConfig(
            name=setup["experiment_name"],
            storage_path=storage_path,
            stop={"timesteps_total": setup["nb_timesteps"]},  # Stop condition for individual trials
            # MaxCustomMetricStopper("total_agent_interact", setup["nb_timesteps"]), #
            # "custom_metrics/grid2op_end_mean": setup["max_ep_len"]},
            callbacks=[
                WandbLoggerCallback(project=setup["experiment_name"]),
                TuneCallback(
                    setup["my_log_level"],
                    "evaluation/custom_metrics/grid2op_end_mean",
                    eval_freq=config["evaluation_interval"],
                    heartbeat_freq=60,
                ),
            ],
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=setup["checkpoint_freq"],
                checkpoint_at_end=True,
                checkpoint_score_attribute="custom_metrics/corrected_ep_len_mean",
                num_to_keep=5,
            ),
            verbose=setup["verbose"],
        ),
        tune_config=tune.TuneConfig(
            trial_name_creator=lambda t: trial_str_creator(t, job_id),
            trial_dirname_creator=lambda t: trial_dir_name(t),
            search_alg=algo,
            num_samples=setup["num_samples"],
            time_budget_s=time_budget,  # Time budget for entire optimization
            # scheduler=scheduler,
        ) if setup["optimize"] else
        tune.TuneConfig(
            trial_name_creator=lambda t: trial_str_creator(t, job_id),
            trial_dirname_creator=lambda t: trial_dir_name(t), )
        ,
    )

    # Launch tuning
    try:
        result_grid = tuner.fit()
    finally:
        # Close ray instance
        ray.shutdown()

    for i in range(len(result_grid)):
        result = result_grid[i]
        if not result.error:
            # Print and save available checkpoints
            checkpoints_tojson = {
                os.path.basename(checkpoint.path): metrics['evaluation']['custom_metrics'] for
                checkpoint, metrics in result.best_checkpoints
            }
            with open(os.path.join(result.path, "checkpoint_results.json"), "w") as outfile:
                json.dump(checkpoints_tojson, outfile)

            print(Style.BOLD + f" *---- Trial {i} finished successfully with evaluation results ---*\n" + Style.END +
                  tabulate(
                      [[k] + list(v.values()) for k, v in checkpoints_tojson.items()],
                      headers=['checkpoint'] + list(result.metrics['evaluation']['custom_metrics'].keys()),
                      tablefmt='rounded_grid')
                  )

            # print("ALL RESULT METRICS: ", result.metrics)
            # print("ENV CONFIG: ", result.configs['env_config'])
            # print("RESULT CONFIG: ", result.configs['env_config'])
            # Print table with environment configs.
            if REPORT_END:
                print(f"--- Environment Configuration  ---- \n"
                      f"{tabulate([result.config['env_config']], headers='keys', tablefmt='rounded_grid')}")
                # print other params:
                params_ppo = ['gamma', 'lr', 'exploration_config', 'vf_loss_coeff', 'entropy_coeff', 'clip_param',
                              'lambda', 'vf_clip_param', 'num_sgd_iter', 'sgd_minibatch_size', 'train_batch_size']
                values = [result.config[par] for par in params_ppo]
                print(f"--- PPO Configuration  ---- \n"
                      f"{tabulate([values], headers=params_ppo, tablefmt='rounded_grid')}")
                params_model = ['fcnet_hiddens', 'fcnet_activation', 'post_fcnet_hiddens', 'post_fcnet_activation']
                values = [result.config['model'][par] for par in params_model]
                print(f"--- Model Configuration  ---- \n"
                      f"{tabulate([values], headers=params_model, tablefmt='rounded_grid')}")
        else:
            print(f"Trial failed with error {result.error}.")

    # If Optuna optimization was enabled, save results summary
    if setup.get("optimize", False):
        save_optuna_results_summary(result_grid, setup, setup.get("workdir", "."))

        # Save the Optuna study to a SQLite database for dashboard access
        if algo is not None:
            try:
                optuna_db_dir = os.path.join(setup.get("workdir", "."), "results", "optuna_studies")
                db_path = algo.save_study(optuna_db_dir, setup['experiment_name'])
                print(f"\n{Style.BOLD}{'='*80}{Style.END}")
                print(f"{Style.BOLD}Optuna study saved to: {db_path}{Style.END}")
                print(f"{Style.BOLD}To view in Optuna Dashboard, run:{Style.END}")
                print(f"  optuna-dashboard sqlite:///{db_path}")
                print(f"{Style.BOLD}{'='*80}{Style.END}\n")
            except Exception as e:
                print(f"Warning: Failed to save Optuna study: {e}")

    return result_grid


def save_optuna_results_summary(result_grid: ResultGrid, setup: dict[str, Any], workdir: str) -> None:
    """
    Save Optuna optimization results to CSV and display summary in terminal.

    Args:
        result_grid: The ResultGrid from Ray Tune containing all trial results
        setup: Setup configuration dictionary
        workdir: Working directory for saving results
    """
    # Prepare data for CSV
    results_data = []

    for i, result in enumerate(result_grid):
        trial_data = {
            'trial_id': i,
            'trial_name': result.path.split('/')[-1] if hasattr(result, 'path') else f"trial_{i}",
            'status': 'SUCCESS' if not result.error else 'FAILED',
        }

        if not result.error and result.metrics:
            # Extract hyperparameters that were optimized
            if 'model' in result.config and 'custom_model_config' in result.config['model']:
                gnn_config = result.config['model']['custom_model_config'].get('gnn', {})
                trial_data['hidden_dim'] = gnn_config.get('hidden_dim', 'N/A')
                trial_data['out_dim'] = gnn_config.get('out_dim', 'N/A')
                trial_data['num_layers'] = gnn_config.get('num_layers', 'N/A')
                trial_data['residual'] = gnn_config.get('residual', 'N/A')

            # Extract key metrics
            if 'evaluation' in result.metrics and 'custom_metrics' in result.metrics['evaluation']:
                custom_metrics = result.metrics['evaluation']['custom_metrics']
                trial_data['grid2op_end_mean'] = custom_metrics.get('grid2op_end_mean', 'N/A')
                trial_data['corrected_ep_len_mean'] = custom_metrics.get('corrected_ep_len_mean', 'N/A')
                trial_data['episode_reward_mean'] = custom_metrics.get('episode_reward_mean', 'N/A')

            # Extract training metrics
            trial_data['timesteps_total'] = result.metrics.get('timesteps_total', 'N/A')
            trial_data['training_iteration'] = result.metrics.get('training_iteration', 'N/A')

        else:
            trial_data['error'] = str(result.error) if result.error else 'Unknown'

        results_data.append(trial_data)

    # Create DataFrame
    df = pd.DataFrame(results_data)

    # Save to CSV
    csv_dir = os.path.join(workdir, "results", "optuna_results")
    os.makedirs(csv_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"{setup['experiment_name']}_{timestamp}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    df.to_csv(csv_path, index=False)
    print(f"\n{Style.BOLD}{'='*80}{Style.END}")
    print(f"{Style.BOLD}Optuna results saved to: {csv_path}{Style.END}")
    print(f"{Style.BOLD}{'='*80}{Style.END}\n")

    # Display summary in terminal
    print(f"\n{Style.BOLD}=== OPTUNA OPTIMIZATION SUMMARY ==={Style.END}\n")
    print(f"Total trials: {len(results_data)}")
    successful_trials = sum(1 for r in results_data if r['status'] == 'SUCCESS')
    print(f"Successful trials: {successful_trials}")
    print(f"Failed trials: {len(results_data) - successful_trials}\n")

    # Display top 5 trials by target metric
    if successful_trials > 0:
        metric_col = setup.get("score_metric", "grid2op_end_mean").split('/')[-1]  # Get last part of metric path

        # Only display if the metric column exists
        if metric_col in df.columns:
            # Convert to numeric, handling 'N/A' values
            df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
            top_trials = df[df['status'] == 'SUCCESS'].nlargest(5, metric_col)

            print(f"{Style.BOLD}Top 5 Trials (by {metric_col}):{Style.END}")
            print(tabulate(top_trials, headers='keys', tablefmt='rounded_grid', showindex=False))
            print()

        # Display all trials summary
        print(f"\n{Style.BOLD}All Trials Summary:{Style.END}")
        display_cols = [col for col in df.columns if col not in ['error', 'trial_name']]
        print(tabulate(df[display_cols], headers='keys', tablefmt='rounded_grid', showindex=False))
        print()

        # Display best hyperparameters
        if metric_col in df.columns:
            metric_series = df[metric_col]
            has_valid_data = bool(not metric_series.isna().all())
            if has_valid_data:
                best_idx = df[df['status'] == 'SUCCESS'][metric_col].idxmax()
                best_trial = df.loc[best_idx]

                print(f"\n{Style.BOLD}Best Trial (Trial {int(best_trial['trial_id'])}):{Style.END}")
                print(f"  {metric_col}: {best_trial[metric_col]}")
                try:
                    if pd.notna(best_trial.get('hidden_dim')):
                        print(f"  Hyperparameters:")
                        print(f"    - hidden_dim: {best_trial['hidden_dim']}")
                        print(f"    - out_dim: {best_trial['out_dim']}")
                        print(f"    - num_layers: {best_trial['num_layers']}")
                        print(f"    - residual: {best_trial['residual']}")
                except (KeyError, AttributeError):
                    pass
                print()

    print(f"{Style.BOLD}{'='*80}{Style.END}")
    print(f"{Style.BOLD}Full results available at: {csv_path}{Style.END}")
    print(f"{Style.BOLD}{'='*80}{Style.END}\n")

